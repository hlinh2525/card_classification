# card-classification

53-class playing card classification (52 standard cards and a joker).

Place the original dataset at `data/raw/dataset.zip`, with images arranged as
`dataset/<class name> <image number>.jpg` (for example `dataset/ace of clubs 01.jpg`).

Run dataset inspection from the project root:

```powershell
python src/data/inspect.py
```

Requires Pillow (`pip install Pillow`). Paths are defined in `configs/config.py`
and resolved relative to the project, independent of the working directory.
The script reads the ZIP directly and writes `clean_metadata.csv`, `train.csv`,
`val.csv`, and `test.csv` into `data/metadata`. The `member` column refers to a
path inside `dataset.zip`, not an extracted file on disk. Class IDs are shared
across splits and assigned in alphabetical order.

Inspection checks readable images and all 53 classes, removes exact byte
duplicates, and excludes images with conflicting labels. It then splits each
class into approximately 70% train, 15% validation, and 15% test with seed 42.
Deduplication happens before splitting to prevent exact duplicates leaking
across splits. These are new splits; the flattened archive has no original split labels.
Image dimensions and color modes are reported; preprocessing should convert
images to RGB and resize them to `IMAGE_SIZE`. Hash checks detect exact file
duplicates only, not visually similar or re-encoded images.
