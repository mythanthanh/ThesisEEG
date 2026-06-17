import re
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne

logger = logging.getLogger(__name__)


class StatusOffsetError(RuntimeError):
    pass

class Preprocessing():
    def __init__(self, eeg_path,
                 bad_channels, detect_muscle_ics=False, report=True, report_path=Path('data/reports/')):
        pass
        self.eeg_path = eeg_path
        self.file_name = eeg_path.stem
        self.bad_channels = bad_channels
        self.detect_muscle_ics = detect_muscle_ics
        self.report = report
        self.report_path = report_path
        report_path.mkdir(parents=True, exist_ok=True)
        self.output_path = Path('data/processed/')
        self.output_path.mkdir(parents=True, exist_ok=True)

    def run(self):
        try:
            raw = mne.io.read_raw_brainvision(
                self.eeg_path,
                preload=True,
                eog=['EOG1', 'EOG2'],
                misc=['ECG']
            )
        except FileNotFoundError:
            print(f"[WARNING] Missing EEG file: {self.eeg_path}. Skipping.")
            return None, None
        except Exception as e:
            print(f"[ERROR] Failed to load {self.eeg_path}: {e}")
            return None, None
        #if self.report:
           #self.report_object = mne.Report(title=f"Report for {self.file_name}")
           #self.report_object.add_raw(raw, title="Raw data overview")
        raw.set_channel_types({'EOG1': 'eog', 'EOG2': 'eog', 'ECG': 'ecg'})
        set_montage(raw)
        line_ratio = bandpass_and_notch(raw)
        mark_bads(raw, self.bad_channels) #TODO: test run and comment out after 
                          #self.report_path,
                          #detect_muscle_ics=self.detect_muscle_ics,
                          #report=self.report)
        ica_result = run_ica(
            raw, 
            self.file_name, 
            self.report_path, 
            detect_muscle_ics=self.detect_muscle_ics, 
            report=self.report
        )
        if ica_result is not None:
           raw = ica_result
        else:
            print(f"Warning: ICA result was None for {self.file_name}. Skipping ICA application.")
        raw.set_eeg_reference('average')
        raw.save(f"{self.output_path}/{self.file_name}_clean_raw.fif", overwrite=True)
        epochs_clean = epoch_and_reject(raw) #TODO: replace with raw_ICA
        if epochs_clean.info['bads']:
            epochs_clean.interpolate_bads()
        epochs_clean.set_eeg_reference('average', projection=True)  # not applying the projection yet, it will be applied later during source estimation
        #if self.report:
           #report_file = Path(self.report_path) / "report.html"
           #self.report_object.save(report_file, overwrite=True)
        return epochs_clean, line_ratio

#def crop_offset(raw, start_time): #do I need to keep this 
   # """Crop the raw data based on film start offsets."""
    # sanity check before cutting the offset 
   # status_ch_events = mne.find_events(raw, stim_channel="Status")

   # if len(status_ch_events) < 2:
       # raise StatusOffsetError(
            #f"Status channel has <2 events for {raw.filenames[0] if raw.filenames else 'unknown file'}"
        #)

    #if status_ch_events[1, 0] == start_time * raw.info['sfreq']:  # the timing of the second event in the satus channel should corresponds to the movie start (where we cut the data)
       # print('The offset is compatible with Status channel')
        #raw.crop(tmin=start_time)
    #else:
        #raise StatusOffsetError(
            #f"Status channel offset mismatch for {raw.filenames[0] if raw.filenames else 'unknown file'} "
            #f"(expected {start_time * raw.info['sfreq']} samples)"
       # )
    
   # return raw

def set_montage(raw, montage='standard_1020'):
    mon = mne.channels.make_standard_montage(montage)
    raw.set_montage(mon)

def bandpass_and_notch(raw, l_freq=1., h_freq=50., notch_freqs=[50.]):
    raw.filter(l_freq, h_freq, picks=["eeg"])
    if notch_freqs:
        raw.notch_filter(notch_freqs, notch_widths=0.02, method="fir", picks="eeg")
        line_ratio = _calculate_line_ratio(raw)
    return line_ratio


def mark_bads(raw, bads):
    if isinstance(bads, float) and np.isnan(bads):
        # no bad channels
        raw.info['bads'] = []
        return

    elif isinstance(bads, str) and bads.startswith('['):
        # stringified list: "['P7', 'T8']"
        raw.info['bads'] = re.findall(r"'([^']+)'", bads)
        return

    elif isinstance(bads, str):
        # single channel: 'O1'
        raw.info['bads'] = [bads]
        return

    elif isinstance(bads, list):
        # already a list
        raw.info['bads'] = [ch for ch in bads if ch]
        return

    else:
        raise ValueError(f"Unexpected bad_channels entry: {bads}")

        
def run_ica(raw, file_name, report_path, detect_muscle_ics=False, report=True): #TODO: replace this with notebook one 
    # Vertical EOG proxy: Fp1 - Cz
    raw = mne.set_bipolar_reference(raw, "Fp1", "Cz", ch_name="VEOG", drop_refs=False)
    raw.set_channel_types({"VEOG": "eog"})

    # Horizontal EOG proxy (optional): F7 - F8
    #use the EOG1 and EOG2 inside the jupyter notebook -done
    raw = mne.set_bipolar_reference(raw, "F7", "F8", ch_name="HEOG", drop_refs=False)
    raw.set_channel_types({"EOG1": "eog", "EOG2": "eog"})

    # lowpass filtered data for ICA fitting
    raw_filt = raw.copy().filter(None, 40., picks=["eeg", "eog"])
    ica = mne.preprocessing.ICA(n_components=0.99, method="fastica", random_state=97)
    ica.fit(raw_filt, picks="eeg", reject_by_annotation=True)

    eog_inds_v, scores_v = ica.find_bads_eog(raw, ch_name="VEOG")
    eog_inds_h, scores_h = ica.find_bads_eog(raw, ch_name="HEOG")
    #muscle_inds, muscle_scores = [], []
    #if detect_muscle_ics:
        #muscle_inds, muscle_scores = ica.find_bads_muscle(raw)
        #muscle_inds = [i for i in muscle_inds if abs(muscle_scores[i]) > 0.9]

    eog_inds_v = [i for i in eog_inds_v if abs(scores_v[i]) >= 0.5]
    eog_inds_h = [i for i in eog_inds_h if abs(scores_h[i]) >= 0.5]

    #bad_ic = sorted(set(eog_inds_v + eog_inds_h + muscle_inds))
    bad_ic = sorted(set(eog_inds_v + eog_inds_h))
    ica.exclude = bad_ic

    raw_ica = ica.apply(raw.copy())
    raw_ica.drop_channels(['VEOG', 'HEOG'])


    if report:
        # add table and a bar plot of scores to the report
        df_scores = pd.DataFrame({
            "IC": np.arange(ica.n_components_),
            "EOG_V_score": scores_v,
            "EOG_H_score": scores_h,
        })
        #if len(muscle_scores):
            #df_scores["Muscle_score"] = muscle_scores

        #_create_ica_report(raw_filt, ica, df_scores, file_name, report_path)

    return raw_ica

def _create_ica_report(raw, ica, df_scores, file_name, report_path):
    report = mne.Report(title=f"ICA report – {file_name}")

    report.add_ica(
        ica=ica,
        title="ICA components",
        inst=raw,
        picks=ica.exclude if ica.exclude else None,
        n_jobs=1
    )

    figs = ica.plot_components(show=False)
    report.add_figure(figs, title="All IC topographies")
    if isinstance(figs, (list, tuple)):
        for fig in figs:
            if isinstance(fig, plt.Figure):
                plt.close(fig)
    else:
        if isinstance(figs, plt.Figure):
            plt.close(figs)

    fig_sources = ica.plot_sources(raw, show=False)
    report.add_figure(fig_sources, title="IC time courses")
    if isinstance(fig_sources, plt.Figure):
        plt.close(fig_sources)

    html = df_scores.to_html(index=False, float_format="%.3f")

    report.add_html(
        html,
        title="ICA EOG and Muscle correlation scores"
    )

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(df_scores["IC"], df_scores["EOG_V_score"], label="VEOG")
    ax.bar(df_scores["IC"], df_scores["EOG_H_score"], alpha=0.6, label="HEOG")
    if hasattr(df_scores, "Muscle_score"):
        ax.bar(df_scores["IC"], df_scores["Muscle_score"], alpha=0.6, label="Muscle")
        ax.axhline(0.9, color="b", linestyle="--", linewidth=1)
    ax.axhline(0.5, color="r", linestyle="--", linewidth=1)
    ax.axhline(-0.5, color="r", linestyle="--", linewidth=1)
    ax.set_xlabel("IC")
    ax.set_ylabel("Correlation")
    ax.legend()
    report.add_figure(fig, title="EOG–IC correlation scores")
    plt.close(fig)

    fig, ax = plt.subplots(1, len(df_scores.columns)-1, figsize=(20, 4))
    ax[0].hist(df_scores["EOG_V_score"], label="VEOG")
    ax[0].set_title("VEGO Scores")
    ax[1].hist(df_scores["EOG_H_score"], label="HEOG")
    ax[1].set_title("HEGO Scores")
    if hasattr(df_scores, "Muscle_score"):
        ax[2].hist(df_scores["Muscle_score"], label="Muscle")
        ax[2].set_title("Muscle Scores")

    report.add_figure(fig, title="Distribution of Scores")
    plt.close(fig)

    report.save(
        report_path / f"{file_name}_ica_report.html",
        overwrite=True,
        open_browser=False
    )

#def set_reference(raw, ref_channels=['average']): #TODO: check if needed 
    #raw.set_eeg_reference(ref_channels) 
def epoch_and_reject(raw): 
        epochs = mne.make_fixed_length_epochs(
            raw,
            duration=1.0, #change to 4.0
            overlap=0.0,
            preload=True
        )
        reject = dict(eeg=150e-6)
        epochs.drop_bad(reject=reject)
        return epochs

def _calculate_line_ratio(raw):
    psd, freqs = mne.time_frequency.psd_array_welch(
        raw.get_data(picks="eeg"), sfreq=512, fmin=1, fmax=60, n_fft=8192, verbose=False
    )
    line_ratio = psd[..., (freqs > 49.5) & (freqs < 50.5)].mean() - psd[..., ((freqs > 48) & (freqs < 49)) | ((freqs > 51) & (freqs < 52))].mean()
    return float(line_ratio)
