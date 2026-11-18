"""Fetch Poly Haven assets through its public API (api.polyhaven.com), each file checked against the API's md5.
  python polyhaven.py --out ./polyhaven --res 4k \
      --models nature,tools,containers,furniture --collections pine_forest,verdant_trail,the_shed,smugglers_cove,hidden_alley \
      --hdris night,overcast --textures brick,cobblestone,bark,roofing,plaster-concrete,rock [--dry]
Models: the FBX at --res with its textures. Textures: every map at --res as jpg (png where there is no jpg). HDRIs: .hdr.
Writes <out>/manifest.jsonl (one line per asset: id, type, categories, files, bytes). Skips a file already on disk with
the right md5, so a rerun resumes. Assets: CC0 (polyhaven.com/license). API terms: polyhaven.com/our-api."""
import argparse, concurrent.futures as cf, hashlib, json, os, sys, urllib.parse, urllib.request

UA = {"User-Agent": "polyhaven-bulk-download/1.0 (+https://github.com/Chris0Jeky/lumenfold-open)"}
API = "https://api.polyhaven.com"


def js(path):
    return json.load(urllib.request.urlopen(urllib.request.Request(API + path, headers=UA), timeout=60))


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def fetch(url, dst, want):
    if os.path.exists(dst) and md5(dst) == want:
        return "have"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r, open(dst + ".part", "wb") as f:
        while b := r.read(1 << 20):
            f.write(b)
    got = md5(dst + ".part")
    if got != want:
        os.remove(dst + ".part")
        return f"MD5 MISMATCH {url}"
    os.replace(dst + ".part", dst)
    return "ok"


def files_for(aid, kind, res):
    """[(url, relative path, md5, size)] for one asset at res, falling back to the largest available below it."""
    f = js(f"/files/{aid}")
    order = ["8k", "4k", "2k", "1k"]
    pick = lambda d: d.get(res) or next((d[r] for r in order[order.index(res) + 1:] if r in d), None)
    out = []
    if kind == "models":
        e = pick(f.get("fbx", {}))
        e = e and e.get("fbx")
        if e:
            out.append((e["url"], os.path.basename(e["url"]), e["md5"], e["size"]))
            out += [(v["url"], k, v["md5"], v["size"]) for k, v in e.get("include", {}).items()]
    elif kind == "hdris":
        e = pick(f.get("hdri", {}))
        e = e and (e.get("hdr") or e.get("exr"))
        if e:
            out.append((e["url"], os.path.basename(e["url"]), e["md5"], e["size"]))
    else:
        for m, d in f.items():
            if m in ("blend", "gltf", "usd", "mtlx") or not isinstance(d, dict):
                continue
            e = pick(d)
            e = e and (e.get("jpg") or e.get("png") or e.get("exr"))
            if e:
                out.append((e["url"], os.path.basename(e["url"]), e["md5"], e["size"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", default="4k", choices=["8k", "4k", "2k", "1k"])
    for k in ("models", "collections", "hdris", "textures"):
        ap.add_argument("--" + k, default="")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    want = {}  # id -> (kind, categories). One call per category: a comma list in the API means "in all of them".
    sel = [(k, c) for k in ("models", "hdris", "textures") for c in getattr(a, k).split(",") if c]
    sel += [(k, "collection: " + c) for c in a.collections.split(",") if c for k in ("models", "textures", "hdris")]
    for kind, cat in sel:
        q = urllib.parse.quote(cat)
        for aid, info in js(f"/assets?type={kind}&categories={q}").items():
            want.setdefault(aid, (kind, info.get("categories", [])))
    print("assets", len(want), {k: sum(1 for v in want.values() if v[0] == k) for k in ("models", "hdris", "textures")},
          flush=True)
    jobs, man = [], []
    for aid, (kind, cats) in sorted(want.items()):
        fs = files_for(aid, kind, a.res)
        man.append({"id": aid, "type": kind, "categories": cats, "files": len(fs), "bytes": sum(x[3] for x in fs)})
        jobs += [(u, os.path.join(a.out, kind, aid, rel), m) for u, rel, m, _ in fs]
    total = sum(x["bytes"] for x in man)
    print("files", len(jobs), "GB", round(total / 1e9, 2), flush=True)
    if a.dry:
        return
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "manifest.jsonl"), "a", encoding="utf-8") as f:
        for m in man:
            f.write(json.dumps(m) + "\n")
    bad = 0
    with cf.ThreadPoolExecutor(4) as ex:
        for (u, dst, m), r in zip(jobs, ex.map(lambda j: fetch(*j), jobs)):
            if r not in ("ok", "have"):
                bad += 1
                print(r, flush=True)
    print("done files", len(jobs), "bad", bad, flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
