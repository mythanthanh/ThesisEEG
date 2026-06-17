import os
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from preprocessingmeditation import Preprocessing

os.environ["MNE_BROWSER_BACKEND"] = "matplotlib"

logging.basicConfig(
    filename="preprocessing.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

def run_preprocessing_pipeline(bad_channels_path, eeg_base_path):
    bad_channels = pd.read_excel(bad_channels_path)
    bad_channels.set_index('subject', inplace=True)

    subjects = [f"{i:03d}" for i in range(1, 98)]  # change to range(1, 98) for all subjects
    base_path = "ds003969-download"

    for sub in subjects:
        eeg_dir = Path(base_path) / f"sub-{sub}" / "eeg"

         # find all BDF files for this subject
        bdf_files = list(eeg_dir.glob(f"sub-{sub}_task-*_eeg.bdf"))
        if not bdf_files:
           print(f"No tasks found for subject {sub}")
           continue

    # get bad channels
        entry = bad_channels.loc[int(sub), 'bad_channels'] if int(sub) in bad_channels.index else None
        bad_ch = [] if entry is None or pd.isna(entry) else [ch.strip() for ch in str(entry).split(',')]

        for bdf_path in bdf_files:
           fname = bdf_path.name
           task = fname.split("_")[1].replace("task-", "")  # extract task name

           report_dir = Path("data/reports") / f"sub-{sub}_task-{task}"
           report_dir.mkdir(parents=True, exist_ok=True)

           preprocessing = Preprocessing(
               eeg_path=bdf_path,
               bad_channels=bad_ch,
               report=True,
               report_path=report_dir
             )

           epochs_clean, line_ratio = preprocessing.run()

           if epochs_clean is None:
               print(f"[SKIP] Subject {sub}, task {task} — no valid EEG data.")
               continue
 
           epochs_dir = Path("data/epochs")
           epochs_dir.mkdir(parents=True, exist_ok=True)
           epochs_clean.save(epochs_dir / f"sub-{sub}_task-{task}_epo.fif", overwrite=True)

           print(f"Subject {sub}, task {task} processed successfully.")


  
if __name__ == "__main__":
    base_directory = Path(r"D:\Thesis\ThesisEEG")
    bad_channels_path = base_directory / "ds003969-download/bad_channels.xlsx"
    eeg_base_path = base_directory / "ds003969-download"
    #os.chdir(base_directory)
    if base_directory.exists():
        os.chdir(base_directory)
    run_preprocessing_pipeline(bad_channels_path, eeg_base_path)

