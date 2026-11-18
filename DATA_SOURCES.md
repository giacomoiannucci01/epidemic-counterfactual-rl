# Data and source provenance

The package contains aggregated national surveillance only, trimmed to the dates/channels needed for initial filtering, historical-fit comparisons and the study period. CSVs are canonical; Excel workbooks expose the same observations for inspection.

| Country/input | Source and meaning |
|---|---|
| England cases, age-specific hospital additions, all-age MV occupancy | The supplied `observations_england.csv` used in the previous analyses, 12 Oct 2020–31 Jan 2021. Admissions are the existing hospital-additions proxy; MV occupancy is not total ICU occupancy. |
| England total deaths | [UKHSA historical dashboard archive](https://ukhsa-dashboard.data.gov.uk/covid-19-archive-data-download), `Deaths/nation_newDeaths28DaysByDeathDate.csv`, England E92000001, deaths within 28 days of a positive test by date of death. |
| England deaths by age | Original UKHSA five-year death bands preserved in the [COVerAGE input database](https://osf.io/mpwjq/), 12 August 2022 snapshot, aggregated to 0–44, 45–64, 65+. This is the original input-count database, not age-harmonised estimates. |
| Belgium cases, admissions, ICU occupancy | The supplied Belgian official-data export based on Sciensano. Case deciles are exactly aggregated to reported 0–39, 40–59, 60+; admissions and ICU remain all-age totals. [Sciensano open-data documentation](https://epistat.sciensano.be/covid/). |
| Belgium deaths | The supplied Sciensano `COVID19BE_MORT.csv`, summed over regions and sexes, with native ages aggregated to 0–44, 45–64, 65+ and a separate unknown-age column. |
| Contact inputs | Original supplied CoMix/POLYMOD reconstructions and government contact calendars, including the 56-day common health tail. These are reconstructed effective contacts, not direct legal-policy magnitudes. |
| Fixed q scenarios | Previously supplied England world-0 genealogy and Belgian V10 152-day scenario; retrospective model inputs, not measured surveillance. |

Source-archive member hashes are in `provenance/parameter_sources.json`. Packaged-file hashes are in `provenance/data_sha256.json`. The legacy source-kernel excerpts are provided for algorithm audit. Source data retain their originating attribution and terms.

No exact rebinning of observed Belgian cases to 0–44/45–64/65+ is asserted. The model-to-observation mapping is population-based and applies to model predictions only. No age-specific hospital/ICU observations are fabricated.
