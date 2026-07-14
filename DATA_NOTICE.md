# Data attribution and license

The source code in this repository is released under the MIT License (see
`LICENSE`). The **data files** in the `Input/` directory are derived from
third-party sources and are subject to separate terms, summarized below.

## Sources

### Flight connectivity (`opensky_el_final.csv`)

The basin-to-basin edge list in this repository is an aggregated and
anonymized derivative of the OpenSky Network COVID-19 flight-list dataset:

> Olive, X., Strohmeier, M., Lübbe, J.
> *Crowdsourced air traffic data from The OpenSky Network 2020.*
> Zenodo. https://doi.org/10.5281/zenodo.7923702

That dataset is itself a cleaned derivative of the full OpenSky historical
archive (ADS-B trajectories).

**Aggregation pipeline.** Starting from the per-flight records in the Zenodo
dataset, the following fields were dropped: `callsign`, `icao24` (transponder
ID), `registration` (tail number), `typecode`, `firstseen`, `lastseen`,
`day`, and the per-flight latitude/longitude/altitude triples. Origin and
destination ICAO airport codes were then mapped to geographic basins
(defined by GLEAMviz — see below) and aggregated. The published file
consists only of unordered pairs `(basin1, basin2)` indicating that at
least one flight was observed between those two basins in the reference
period. No individual flight, aircraft, route, or timestamp can be
reconstructed from this file.

**License.** The Zenodo dataset is distributed under the OpenSky Network
General Terms of Use & Data License Agreement. The full license text is
bundled with the Zenodo record and is also available here:
https://opensky-network.org/about/terms-of-use

Key obligations for any downstream user of *this* aggregated derivative:

1. **Citation (Section 4(i) of the license).** You must cite the OpenSky
   Network as a data source, either by URL:

   > The OpenSky Network, https://www.opensky-network.org

   or by referencing the founding paper:

   > Matthias Schäfer, Martin Strohmeier, Vincent Lenders, Ivan Martinovic,
   > Matthias Wilhelm. *Bringing Up OpenSky: A Large-scale ADS-B Sensor
   > Network for Research.* Proceedings of the 13th IEEE/ACM International
   > Symposium on Information Processing in Sensor Networks (IPSN), 2014,
   > pp. 83–94.

   Additionally, please cite the specific Zenodo dataset:

   > Strohmeier, M., Olive, X., Lübbe, J., Schäfer, M., Lenders, V.
   > *Crowdsourced air traffic data from the OpenSky Network 2019–2020.*
   > Earth System Science Data 13(2), 2021.
   > https://doi.org/10.5194/essd-13-357-2021

2. **Notification.** If you publish work using this data, OpenSky Network
   requires that you send them a link to the publication
   (contact@opensky-network.org).

3. **Scope.** Use is limited to non-profit research, non-profit education,
   commercial internal testing and evaluation, or government purposes.
   Other commercial uses require a separate license from OpenSky.

4. **No re-identification.** You may not attempt to reverse-engineer,
   de-anonymize, or re-identify the records.

5. **Raw data is NOT redistributed here.** Users who need the unaggregated
   per-flight records should download them directly from Zenodo at
   https://doi.org/10.5281/zenodo.7923702 and accept the OpenSky Terms of
   Use directly with OpenSky Network.

### Basin geography (`basin_id`, `regions`, `latitude`, `longitude` in `opensky_nodes_final.csv`)

The basin partitioning scheme — the definition of each basin, its
identifier, its representative latitude/longitude, and its region label —
comes from the **GLEAMviz** global epidemic and mobility model:

> Broeck, W. Van den, Gioannini, C., Gonçalves, B., Quaggiotto, M.,
> Colizza, V., Vespignani, A.
> *The GLEaMviz computational tool, a publicly available software to
> explore realistic epidemic spreading scenarios at the global scale.*
> BMC Infectious Diseases 11, 37 (2011).
> https://doi.org/10.1186/1471-2334-11-37

More information: http://www.gleamviz.org/

Users who want to work with the full GLEAMviz basin scheme or reproduce
the mapping from ICAO airport codes to basins should obtain the basin
data directly from GLEAMviz. The subset included here (basin IDs,
coordinates, region labels, and Wikipedia-derived passenger strengths
for basins that appear in our reconstruction) is redistributed as a
reproducibility artifact for this specific paper; downstream uses that
require the full scheme should consult GLEAMviz directly.

### Passenger volumes (`pax_strength` column in `opensky_nodes_final.csv`)

Airport-level annual passenger figures are sourced from **Wikipedia** and
are made available under the
[Creative Commons Attribution-ShareAlike 4.0 International License
(CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/), as per
Wikipedia's standard licensing terms.

### Fit coefficients (hard-coded in `strength_flows_computation.py`)

The α and β power-law coefficients used in the reconstruction were
estimated by the authors from a proprietary airport-pair traffic dataset
that is **not redistributable** and is not included in this repository.
Only the two fitted numerical values are published here, both inline in
the source code and in the accompanying paper (Fischetti et al.,
arXiv:2601.13867). Uncertainty estimates and R² of the fit are recorded
as comments in the source code.

## Summary

| Component | Origin | Terms |
|-----------|--------|-------|
| Source code | This repository | MIT License (see `LICENSE`) |
| Basin-to-basin edge list | Aggregated derivative of OpenSky / Zenodo 7923702 | OpenSky Terms of Use — attribution and notification required |
| Basin definitions and coordinates | GLEAMviz | Attribution to Van den Broeck et al. (2011) required |
| Passenger volumes | Wikipedia | CC BY-SA 4.0 |
| Fit coefficients | Computed by the authors from a proprietary dataset (raw data not redistributed) | MIT License (numerical values only) |

If you redistribute any derivative of this dataset, please preserve this
notice and the attribution requirements above.
