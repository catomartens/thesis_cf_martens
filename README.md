# UAV Corridor Routing – Thesis Repository

This repository contains the full code and data pipeline developed for the MSc thesis *"Navigating through Very Low Level Airspace"*, conducted at TU Delft in collaboration with Schuberg Philis. The project explores the design and evaluation of fixed UAV delivery corridors in the Netherlands using geospatial data, risk modeling, and graph-based routing.

The full thesis report can be accessed via the TU Delft repository:  
https://repository.tudelft.nl/person/Person_0192ad10-c77e-4c02-8004-e587e91957d4

## Overview

The thesis responds to the growing need for sustainable and efficient last-mile logistics. While UAVs promise operational, environmental, and societal benefits, their deployment in densely regulated and spatially constrained environments like the Netherlands is not straightforward. This project develops and validates a comprehensive framework for UAV corridor planning that:

- Integrates spatial constraints, regulatory compliance (U-space), and operational feasibility  
- Applies a Semi-Quantitative Risk Assessment (SQRA) methodology across five external risk factors and three consequence domains  
- Uses open geospatial data (OSM, government datasets) and PostNL delivery infrastructure as test cases  
- Implements a graph-based routing model that supports corridor-level trade-offs between risk and efficiency using a tunable parameter α

## Repository Structure

thesis_cf_martens/
│
├── 1.get_osm_data/           # Retrieves and maps relevant infrastructure from OSM
├── 2.risk_analysis/          # Assigns risk scores and overlays no-fly zones
├── 3._no_fly_zones/          # 
├── distribution_centres/     # Cleans and prepares distribution centre data
├── hard_constraints/         # Experiments with strict (hard) no-fly zone constraints
├── model/                    # Contains graph creation and pathfinding logic
├── sensitivity_analysis/     # Contains three different sensitivity analysis experiments
│
├── data/                     # Input and reference data (e.g., risk scores, boundaries)
├── output/                   # GeoJSON and result exports
├── requirements.txt          # Python package dependencies
└── README.md


## Notebooks and Workflow

The core workflow consists of the following steps:

1. **Data Retrieval**  
   `get_data.ipynb` — Collects linear infrastructure and corridor data from OpenStreetMap based on a defined boundary.

2. **Risk Assessment**  
   - `assign_risk.ipynb` — Applies the SQRA model using a risk score file (`risk_scores.csv`)

3. **No Fly zones**
   - `assign_no_fly_zones.ipynb` — Integrates no-fly zones from Dutch government and Natura2000 datasets

4. **Graph Construction (in Model)**  
   `get_graph.ipynb` — Converts the cleaned GeoDataFrame into a directed graph suitable for corridor routing

5. **Route Planning (in Model)**  
   `find_paths.ipynb` — Computes UAV paths from distribution centres to PostNL pickup points based on risk–efficiency trade-offs

## Key Features

- **Multi-infrastructure routing:** Integrates roads, railways, waterways, and natural corridors
- **U-space compliance:** Framework supports corridor-level risk classification and geo-awareness
- **Adjustable objective function:** Weighting parameter α allows for flexible prioritization between safety (α=1) and energy efficiency (α=0)
- **Spatial filtering:** Excludes areas above 120 meters and filters temporary or non-binding no-fly zones
- **Manual overrides:** Includes logic for handling edge cases such as disconnected distribution centers

## Dependencies

This project runs on Python 3.9+ and uses the following key libraries:

- `geopandas`, `pandas`, `osmnx`, `shapely`, `networkx`, `matplotlib`

Install with:
pip install -r requirements.txt

## Research Background
The project builds upon the Boxslot Lane concept by Schuberg Philis, which envisions corridor-based UAV delivery with guaranteed connectivity rather than full-area coverage. The thesis operationalizes this concept through a replicable and adaptable routing framework that balances policy relevance, technical feasibility, and spatial realism.

The resulting methodology supports policymakers and infrastructure planners in designing scalable, low-risk UAV networks aligned with both European regulation and Dutch airspace constraints.

## Limitations
- Altitude and energy models are simplified and not yet suitable for full 3D optimization
- Risk parameters rely on structured assumptions rather than empirical incident data
- Grid-based routing may oversimplify natural flight paths
- Some data sources (e.g., OSM, PostNL) contain known inaccuracies

