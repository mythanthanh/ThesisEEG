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
    def __init__(self, eeg_path, #check if it will be eeg path or bdf path 
                 bad_channels, report=True, report_path=Path('datameditation/reports/')):
        pass
        self.eeg_path = eeg_path 
        self.file_name = eeg_path.stem
        self.bad_channels = bad_channels
        self.report = report
        self.report_path = report_path
        report_path.mkdir(parents=True, exist_ok=True)
        self.output_path = Path('datameditation/processed/')
        self.output_path.mkdir(parents=True, exist_ok=True)

    def run(self):
        try:
            raw = mne.io.read_raw_bdf(
                self.eeg_path,
                preload=True,
                eog=['EXG1', 'EXG2', 'EXG3', 'EXG4'],
                misc=['EXG7'] #TODO: consider dropping this 
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
        if self.bad_channels:
            valid_bads = [ch for ch in self.bad_channels if ch in raw.ch_names]
            invalid_bads = [ch for ch in self.bad_channels if ch not in raw.ch_names]
            if invalid_bads:
                print(f"Warning: The following bad channels for {self.file_name} were not found in the data and will be ignored: {invalid_bads}")
            self.bad_channels = valid_bads
            if valid_bads:
                print(f"Marking bad channels for {self.file_name}: {valid_bads}")
                raw.info['bads'] = valid_bads
            else:
                raw.info['bads'] = []
        physio_chs = {'GSR1', 'GSR2', 'Erg1', 'Erg2', 'Resp', 'Plet', 'Temp', 'EXG7'}
        always_drop = {'Status'}
        to_drop = [ch for ch in physio_chs.union(always_drop) if ch in raw.ch_names]
        if to_drop:
              print(f"Dropping non-EEG channels: {to_drop}")
              raw.drop_channels(to_drop)
        else:
              print("No non-EEG channels to drop.")
        if "EXG8" in raw.ch_names:
           print("EXG8 detected — using it as replacement for Fp1")
           if "Fp1" in raw.ch_names:
                raw.drop_channels("Fp1")
           raw.rename_channels({"EXG8": "Fp1"})
           raw.set_channel_types({"Fp1": "eeg"})
        type={}
        map={}
        if "EXG5" in raw.ch_names:
           map["EXG5"] = "M1"
           type["M1"] = "eeg"
        if "EXG6" in raw.ch_names:
           map["EXG6"] = "M2"
           type["M2"] = "eeg"
        if map:
           print(f"Renaming channels: {map}")
           raw.rename_channels(map)
           raw.set_channel_types(type)
    
        set_montage(raw)
        line_ratio = bandpass_and_notch(raw)
        #mark_bads(raw, self.bad_channels) #TODO: test run and comment out after 
                          #self.report_path,
                          #report=self.report)
        ica_result = run_ica(
            raw, 
            self.file_name, 
            self.report_path,  
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


def set_montage(raw, montage='standard_1020'):
    mon = mne.channels.make_standard_montage(montage)
    raw.set_montage(mon)

def bandpass_and_notch(raw, l_freq=1., h_freq=50., notch_freqs=[50.]):
    raw.filter(l_freq, h_freq, picks=["eeg"])
    if notch_freqs:
        raw.notch_filter(notch_freqs, notch_widths=0.02, method="fir", picks="eeg")
        line_ratio = _calculate_line_ratio(raw)
    return line_ratio


#def mark_bads(raw, bads):
    #if isinstance(bads, float) and np.isnan(bads):
        # no bad channels
        #raw.info['bads'] = []
        #return

    #elif isinstance(bads, str) and bads.startswith('['):
        # stringified list: "['P7', 'T8']"
        #raw.info['bads'] = re.findall(r"'([^']+)'", bads)
        #return

    #elif isinstance(bads, str):
        # single channel: 'O1'
        #raw.info['bads'] = [bads]
        #return

    #elif isinstance(bads, list):
        # already a list
        #raw.info['bads'] = [ch for ch in bads if ch]
        #return

    #else:
        #raise ValueError(f"Unexpected bad_channels entry: {bads}")
    
def run_ica(raw, file_name, report_path, report=True):#TODO: replace this with notebook one 
    # Vertical EOG proxy: Fp1 - Cz
    if len(raw.info['bads']) > 0:
        raw.info['bads'] = [ch for ch in raw.info['bads'] if ch in raw.ch_names]
    raw = mne.set_bipolar_reference(raw, "Fp1", "Cz", ch_name="VEOG", drop_refs=False)
    raw.set_channel_types({ 
        'EXG3': 'eog', 'EXG4': 'eog'
          })

    # Horizontal EOG proxy (optional): F7 - F8
    #use the EOG1 and EOG2 inside the jupyter notebook -done
    raw = mne.set_bipolar_reference(raw,"F7", "F8", ch_name="HEOG", drop_refs=False)
    raw.set_channel_types({"EXG1": "eog", "EXG2": "eog"})

    # lowpass filtered data for ICA fitting
    raw_filt = raw.copy().filter(None, 40., picks=["eeg"])
    ica = mne.preprocessing.ICA(n_components=20, method="fastica", random_state=97)
    ica.fit(raw_filt, picks="eeg", reject_by_annotation=False)

    eog_inds_v, scores_v = ica.find_bads_eog(raw, ch_name="VEOG")
    eog_inds_h, scores_h = ica.find_bads_eog(raw, ch_name="HEOG")

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


def epoch_and_reject(raw): 
        epochs = mne.make_fixed_length_epochs(
            raw,
            duration=1.0, #change to 4.0
            overlap=0.0,
            preload=True
        )
        reject = None
        epochs.drop_bad(reject=reject)
        return epochs

def _calculate_line_ratio(raw):
    psd, freqs = mne.time_frequency.psd_array_welch(
        raw.get_data(picks="eeg"), sfreq=512, fmin=1, fmax=60, n_fft=8192, verbose=False
    )
    line_ratio = psd[..., (freqs > 49.5) & (freqs < 50.5)].mean() - psd[..., ((freqs > 48) & (freqs < 49)) | ((freqs > 51) & (freqs < 52))].mean()
    return float(line_ratio)
