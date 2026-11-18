# Poly Haven bulk download

## Problem

Getting a large part of Poly Haven (models, textures and HDRIs) by hand means thousands of clicks, no check that each
file arrived whole, and no clean way to resume a download that stopped part way.

## Who this concerns

Anyone who wants many Poly Haven assets at one resolution on disk. Python 3.8 or later, standard library only, any OS.
Poly Haven assets are CC0: https://polyhaven.com/license

## What we changed

`polyhaven.py` uses Poly Haven's public API (https://api.polyhaven.com):

- selects assets by category or collection, one API call per category (a comma list in the API means "in all of
  them", so the script never sends one);
- for each asset, picks the files at `--res` (falling back to the largest lower resolution): models as FBX with their
  textures, textures as every map in jpg (png where there is no jpg), HDRIs as `.hdr`;
- downloads with four threads to a `.part` file and keeps it only when its md5 matches the API's;
- skips a file already on disk with the right md5, so a rerun resumes;
- appends one line per asset to `<out>/manifest.jsonl`: id, type, categories, file count, bytes.

It sends the User-Agent `polyhaven-bulk-download/1.0 (+https://github.com/Chris0Jeky/lumenfold-open)`. Poly Haven asks
API users to identify their project in the User-Agent; see its API terms at https://polyhaven.com/our-api and the API
documentation at https://github.com/Poly-Haven/Public-API. If you build on this script, put your own project's name and
URL there.

## Result

Our selection (models, collections, night and overcast HDRIs and texture categories, as in the example below) came to
1,016 assets, 5,578 files and 64.1 GB at 4K. The check is the md5 of every file against the API's value; no mismatch
count was recorded.

## How to run

Prerequisites: Python 3.8 or later. No packages. The offline test needs nothing else:

```
python3 -m unittest test_polyhaven -v
```

Expected: `Ran 7 tests` and `OK`. It checks the md5 download, resume and mismatch handling through `file://` URLs, and
the per-kind file choice with a fixed API answer. It needs no network.

Count first with `--dry` (it calls the API but downloads nothing), then download:

```
python3 polyhaven.py --out ./polyhaven --res 4k \
    --models nature,tools,containers,furniture \
    --collections pine_forest,verdant_trail,the_shed,smugglers_cove,hidden_alley \
    --hdris night,overcast --textures brick,cobblestone,bark,roofing,plaster-concrete,rock --dry
```

The dry run prints `assets <n> {...}` and `files <n> GB <size>`. A download ends with `done files <n> bad <k>` and
exits 1 if any file failed its md5 (rerun to retry those). Files land in `<out>/<models|hdris|textures>/<asset id>/`.

## Measured on

Windows 11, Python 3. Network speed and the exact Python version were not recorded.

## Not verified

- The counts above are from our notes of one run; we did not rerun the selection for this release, and Poly Haven's
  catalogue changes over time.
- That every file of the selection downloaded: a complete download was not verified.
- The links to Poly Haven's licence, API terms and API documentation were not re-opened for this release.
- The manifest is appended on every run, so a rerun adds duplicate lines.
- `--res` accepts 8k, 4k, 2k and 1k only; larger HDRI resolutions are not offered.
