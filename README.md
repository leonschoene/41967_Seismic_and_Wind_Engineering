# Seismic and Wind Engineering
The uploaded codes contain lines of code where files are loaded that are not in the repository. In general, the repository do not contain the entire folder structure. To use the codes without problem, you need the following folders:\
../your/path/\
- input
- output
- output_tables
- plots
- plots_wind
- src

The **input** folder contains a file with seismic ground motion data that can be used for time history analysis and the Duhamel integration, and for the wind part, an input file with pressure coefficients. Following the code needs to adjusted since it is currently for an input with 18 columns (pressure tabs) and 12 blocks for each storm event with 4096 lines (pressure coefficients).