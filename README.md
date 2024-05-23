# batch-nmr
### In order to Use this code, please ensure your data is structured as follows:

```bash
file_dir/
├── sample name/condition (1)
├    ├── key (1)
├    └── key (2)
├── sample name/condition (2)
├    ├── key (1)
├    └── key (2)
├── sample name/condition (3)
├    ├── key (1)
├    └── key (2)
└──sample name/condition (4)
     ├── key (1)
     └── key (2)
```

## For Example:
```bash
zt_visualize/
├──  Brain organoid - Control
├    ├── lipid
├    └── water (2)
└──Brain organoid - microglia (2)
     ├── lipid
     └── water

dependencies:
pip install git+https://github.com/NMRPy/nmrpy.git@4aeb0b738b72743900b45cfc9e7f8caaa3381b20
pip install pywt
