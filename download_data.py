"""
Downloads public medical PDFs from PubMed Central.
These are open-access, direct downloads — no browser session needed.
Run: python download_data.py
"""
import os
import urllib.request

docs = [
    {
        "name": "keytruda_trial.pdf",
        "url": "https://www.nejm.org/doi/pdf/10.1056/NEJMoa1606774",
        "desc": "Pembrolizumab (Keytruda) KEYNOTE-024 trial — NEJM"
    },
    {
        "name": "ozempic_trial.pdf",
        "url": "https://www.nejm.org/doi/pdf/10.1056/NEJMoa1607141",
        "desc": "Semaglutide (Ozempic) SUSTAIN-6 trial — NEJM"
    },
]

# Better: use PMC open-access direct PDF links
pmc_docs = [
    {
        "name": "pembrolizumab_review.pdf",
        "url": "https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC9293739&blobtype=pdf",
        "desc": "Pembrolizumab clinical review — PMC open access"
    },
    {
        "name": "semaglutide_review.pdf",
        "url": "https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC8521973&blobtype=pdf",
        "desc": "Semaglutide clinical outcomes — PMC open access"
    },
    {
        "name": "immunotherapy_tables.pdf",
        "url": "https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC7354004&blobtype=pdf",
        "desc": "Cancer immunotherapy adverse events — PMC open access"
    },
]

os.makedirs("data", exist_ok=True)

headers = {
    "User-Agent": "Mozilla/5.0 (compatible; research/1.0)"
}

print("Downloading medical PDFs from PubMed Central...\n")
for doc in pmc_docs:
    path = f"data/{doc['name']}"
    print(f"  Downloading {doc['name']} ...")
    print(f"    {doc['desc']}")
    try:
        req = urllib.request.Request(doc["url"], headers=headers)
        with urllib.request.urlopen(req, timeout=30) as r:
            content = r.read()
        # Verify it's actually a PDF
        if content[:4] == b"%PDF":
            with open(path, "wb") as f:
                f.write(content)
            size_mb = len(content) / (1024 * 1024)
            print(f"    ✓ Saved {size_mb:.1f} MB\n")
        else:
            print(f"    ✗ Got HTML instead of PDF — skipping\n")
    except Exception as e:
        print(f"    ✗ Failed: {e}\n")

print("Done. Check data/ folder:")
os.system("ls -lh data/")