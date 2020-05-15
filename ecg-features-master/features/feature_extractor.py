"""
feature_extractor.py
--------------------
This module provides a class and methods for pre-processing ECG signals and generating feature vectors using feature
extraction libraries.
--------------------
By: Sebastian D. Goodfellow, Ph.D., 2017
"""

# Compatibility imports
from __future__ import absolute_import, division, print_function

# 3rd party imports
import os
import time
import pandas as pd
import scipy.io as sio
from biosppy.signals import ecg
from biosppy.signals.tools import filter_signal

# Local imports
# from features.rri_features import *
# from features.template_features import *
from features.full_waveform_features import *


class Features:

    def __init__(self, ecgarray, fs, feature_groups, leads=None, labels=None):

        # Set parameters
        self.ecgarray = ecgarray
        if leads is None:
            self.leads = [i for i in range(len(ecgarray[0]))]
        else: self.leads = leads
        self.fs = fs
        self.feature_groups = feature_groups
        self.labels = labels

        # Set attributes
        self.features = None

    def get_features(self):
        return self.features

    def extract_features(self, filter_bandwidth, n_signals=None, show=False, labels=None,
                         normalize=True, polarity_check=True, template_before=0.2, template_after=0.4):

        # Create empty features DataFrame
        self.features = pd.DataFrame()

        # Loop through .mat files
        for idx, patient in enumerate(self.ecgarray):
            patient = patient[self.leads,:]

            try:

                # Get start time
                t_start = time.time() 
                
                a = pd.Series({'patientIdx':idx})
                
                for lead, signal_raw in zip(self.leads, patient):

                    # Preprocess signal
                    ts, signal_raw, signal_filtered, rpeaks, templates_ts, templates = self._preprocess_signal( 
                        signal_raw=signal_raw, filter_bandwidth=filter_bandwidth, normalize=normalize,
                        polarity_check=polarity_check, template_before=template_before, template_after=template_after
                    )

                    # Extract features from waveform
                    features = self._group_features(lead = lead, ts=ts, signal_raw=signal_raw,
                                                    signal_filtered=signal_filtered, rpeaks=rpeaks,
                                                    templates_ts=templates_ts, templates=templates,
                                                    template_before=template_before, template_after=template_after)
                    a = a.append(features)
                    

                # Append feature vector
                self.features = self.features.append(a, ignore_index=True)

                # Get end time
                t_end = time.time()

                # Print progress
                if show:
                    print('Finished extracting features from ' + str(idx) + 'th patient | Extraction time: ' +
                          str(np.round((t_end - t_start) / 60, 3)) + ' minutes')

            except ValueError:
                print('Error loading ' + str(idx) + 'th patient')

        # Add labels
        self._add_labels(labels=labels)

    def _add_labels(self, labels):
        """Add label to feature DataFrame."""
        if labels is not None:
            self.features = pd.merge(labels, self.features, on='patientIdx')

    def _preprocess_signal(self, signal_raw, filter_bandwidth, normalize, polarity_check, 
                           template_before, template_after):

        # Filter signal
        signal_filtered = self._apply_filter(signal_raw, filter_bandwidth)

        # Get BioSPPy ECG object
        ecg_object = ecg.ecg(signal=signal_raw, sampling_rate=self.fs, show=False)

        # Get BioSPPy output
        ts = ecg_object['ts']          # Signal time array
        rpeaks = ecg_object['rpeaks']  # rpeak indices

        # Get templates and template time array
        templates, rpeaks = self._extract_templates(signal_filtered, rpeaks, template_before, template_after)
        templates_ts = np.linspace(-template_before, template_after, templates.shape[1], endpoint=False)

        # Polarity check
        signal_raw, signal_filtered, templates = self._check_waveform_polarity(polarity_check=polarity_check,
                                                                               signal_raw=signal_raw,
                                                                               signal_filtered=signal_filtered,
                                                                               templates=templates)
        # Normalize waveform
        signal_raw, signal_filtered, templates = self._normalize_waveform_amplitude(normalize=normalize,
                                                                                    signal_raw=signal_raw,
                                                                                    signal_filtered=signal_filtered,
                                                                                    templates=templates)
        return ts, signal_raw, signal_filtered, rpeaks, templates_ts, templates

    @staticmethod
    def _check_waveform_polarity(polarity_check, signal_raw, signal_filtered, templates): 

        """Invert waveform polarity if necessary."""
        if polarity_check:

            # Get extremes of median templates
            templates_min = np.min(np.median(templates, axis=1))
            templates_max = np.max(np.median(templates, axis=1))

            if np.abs(templates_min) > np.abs(templates_max):
                return signal_raw * -1, signal_filtered * -1, templates * -1
            else:
                return signal_raw, signal_filtered, templates

    @staticmethod
    def _normalize_waveform_amplitude(normalize, signal_raw, signal_filtered, templates): 
        """Normalize waveform amplitude by the median R-peak amplitude."""
        if normalize:

            # Get median templates max
            templates_max = np.max(np.median(templates, axis=1))

            return signal_raw / templates_max, signal_filtered / templates_max, templates / templates_max

    def _extract_templates(self, signal_filtered, rpeaks, before, after): 

        # convert delimiters to samples
        before = int(before * self.fs)
        after = int(after * self.fs)

        # Sort R-Peaks in ascending order
        rpeaks = np.sort(rpeaks)

        # Get number of sample points in waveform
        length = len(signal_filtered)

        # Create empty list for templates
        templates = []

        # Create empty list for new rpeaks that match templates dimension
        rpeaks_new = np.empty(0, dtype=int)

        # Loop through R-Peaks
        for rpeak in rpeaks:

            # Before R-Peak
            a = rpeak - before
            if a < 0:
                continue

            # After R-Peak
            b = rpeak + after
            if b > length:
                break

            # Append template list
            templates.append(signal_filtered[a:b])

            # Append new rpeaks list
            rpeaks_new = np.append(rpeaks_new, rpeak)

        # Convert list to numpy array
        templates = np.array(templates).T

        return templates, rpeaks_new

    def _apply_filter(self, signal_raw, filter_bandwidth): 
        """Apply FIR bandpass filter to waveform."""
        signal_filtered, _, _ = filter_signal(signal=signal_raw, ftype='FIR', band='bandpass',
                                              order=int(0.3 * self.fs), frequency=filter_bandwidth,
                                              sampling_rate=self.fs)
        return signal_filtered

    def _group_features(self, lead, ts, signal_raw, signal_filtered, rpeaks, 
                        templates_ts, templates, template_before, template_after):

        """Get a dictionary of all ECG features"""

        # Empty features dictionary
        features = dict()

        # Loop through feature groups
        for feature_group in self.feature_groups:

            # Full waveform features
            if feature_group == 'full_waveform_features':

                # Extract features
                full_waveform_features = FullWaveformFeatures(lead=lead, ts=ts, signal_raw=signal_raw, 
                                                              signal_filtered=signal_filtered, rpeaks=rpeaks,
                                                              templates_ts=templates_ts, templates=templates,
                                                              fs=self.fs)
                full_waveform_features.extract_full_waveform_features()

                # Update feature dictionary
                features.update(full_waveform_features.get_full_waveform_features())

        return pd.Series(data=features)
