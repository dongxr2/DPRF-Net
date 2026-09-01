# GitHub release checklist

Before making the repository public:

- [ ] Replace placeholder repository description and add the final paper title.
- [ ] Add author names, affiliations, contact information, ORCID identifiers, and paper DOI.
- [ ] Choose and add a software license. This package intentionally does not make that legal choice for the author.
- [ ] Confirm whether the target journal permits public posting of the accepted manuscript.
- [ ] Keep the ESTOGU attribution and CC BY 4.0 notice.
- [ ] Run `python -m pytest -q`.
- [ ] Run the DPRF-Net smoke command from the README.
- [ ] Check `git status` and ensure raw archives, checkpoints, credentials, and private documents are absent.
- [ ] Consider publishing large raw data only through Zenodo; do not commit the 13 GB archive.
- [ ] Add the final BibTeX entry after publication.
