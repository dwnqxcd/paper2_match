# -*- coding: utf-8 -*-
"""Verify the reference list against Crossref. Prints one line per entry."""
import re, json, time, requests, difflib

s = open("paper/build_manuscript_v8.py", encoding="utf-8").read()
blk = s[s.index("refs = ["):s.index("\n]", s.index("refs = ["))]
items = re.findall(r'"(.+?)",?\n', blk + "\n")
assert len(items) == 27, len(items)


def norm(t):
    return re.sub(r"[^a-z0-9 ]", "", t.lower())


out = []
for k, ref in enumerate(items, 1):
    body = re.sub(r"\s*doi:/S+$", "", ref.strip()).rstrip(".")
    q = {"query.bibliographic": body[:300], "rows": 3,
         "select": "title,DOI,container-title,volume,page,issued,type"}
    try:
        r = requests.get("https://api.crossref.org/works", params=q,
                         headers={"User-Agent": "ref-check/1.0 (mailto:tbd@example.com)"}, timeout=20)
        js = r.json()["message"]["items"]
    except Exception as e:
        out.append((k, ref, "ERR " + str(e)[:50])); continue
    best, bs = None, 0
    for it in js:
        t = (it.get("title") or [""])[0]
        sc = difflib.SequenceMatcher(None, norm(t)[:80], norm(body)[:80]).ratio()
        if sc > bs:
            bs, best = sc, it
    if not best:
        out.append((k, ref, "no match")); continue
    y = (best.get("issued", {}).get("date-parts") or [[None]])[0][0]
    out.append((k, ref, dict(score=round(bs, 2), title=(best.get("title") or [""])[0],
                             doi=best.get("DOI"), year=y, vol=best.get("volume"),
                             page=best.get("page"),
                             cont=(best.get("container-title") or [""])[0])))
    time.sleep(0.3)

json.dump(out, open("results/raw/_refcheck.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
for k, ref, m in out:
    if isinstance(m, str):
        print(f"[{k}] {m}"); continue
    print(f"[{k}] score={m['score']} year={m['year']} vol={m['vol']} pp={m['page']}")
    print(f"    ours: {ref[:120]}")
    print(f"    cr  : {m['title'][:90]} | {m['cont'][:45]} | doi:{m['doi']}")
