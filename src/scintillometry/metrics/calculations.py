"""Copyright 2023 Scintillometry-Tools Contributors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

=====

Calculates metrics from datasets.
"""

import pandas as pd

import scintillometry.backend.constructions
import scintillometry.backend.derivations
import scintillometry.backend.iterations
import scintillometry.backend.transects
import scintillometry.visuals.plotting
from scintillometry.backend.constructions import ProfileConstructor


class MetricsTopography:
    """Calculate metrics for topographical data."""

    def __init__(self):
        super().__init__()

    def get_z_params(self, user_args, transect):
        """Get effective and mean path heights of transect.

        Computes effective and mean path heights of transect under
        stable and unstable conditions, and with no height dependency.
        Prints the effective and mean path height for the user-selected
        stability conditions.

        Select the stability conditions using ``--r, --regime <str>``.

        Args:
            user_args (argparse.Namespace): Namespace of user arguments.
            transect (pd.DataFrame): Parsed path transect data.

        Returns:
            dict[float, float]: Tuples of effective and mean path height
            |z_eff| and |z_mean| [m], with stability conditions as keys.
        """

        z_params = scintillometry.backend.transects.get_all_z_parameters(
            path_transect=transect
        )

        scintillometry.backend.transects.print_z_parameters(
            z_eff=z_params[str(user_args.regime)][0],
            z_mean=z_params[str(user_args.regime)][1],
            stability=user_args.regime,
        )

        return z_params


class MetricsFlux:
    """Calculate metrics for fluxes."""

    def __init__(self):
        super().__init__()

    def construct_flux_dataframe(self, user_args, interpolated_data, z_eff):
        """Compute sensible heat flux for free convection.

        Computes sensible heat flux for free convection, plots a
        comparison between the computed flux and the flux recorded by
        the scintillometer, and saves the figure to disk.

        Warning: this will overwrite existing values for |CT2| in
        <interpolated_data>.

        Args:
            user_args (argparse.Namespace): Namespace of user arguments.
            interpolated_data (pd.DataFrame): Dataframe containing
                parsed and localised weather and scintillometer data
                with matching temporal resolution.
            z_eff (np.floating): Effective path height, z_eff| [m].

        Returns:
            pd.DataFrame: Interpolated dataframe with additional column
            for sensible heat flux under free convection, and derived
            values for  |CT2| [|K^2m^-2/3|].
        """

        flux_data = scintillometry.backend.derivations.compute_fluxes(
            input_data=interpolated_data,
            effective_height=z_eff[0],
            beam_params=(user_args.beam_wavelength, user_args.beam_error),
        )

        return flux_data

    def get_nearest_time_index(self, data, time_stamp):
        """Get timestamp of dataframe index nearest to input timestamp.

        Args:
            data (pd.DataFrame): Dataframe with DatetimeIndex.
            time_stamp (pd.Timestamp): Time of day.

        Returns:
            pd.Timestamp: Closest index to time stamp.
        """

        nearest_index = data.index.get_indexer([time_stamp], method="nearest", limit=1)
        loc_idx = data.index[nearest_index]

        return loc_idx[0]

    def append_vertical_variables(self, data):
        """Derives vertical measurements and appends them to input.

        For several days or more of data, the returned dictionary may be
        quite large.

        Args:
            data (dict): Must at least contain the keys "vertical" and
                "weather", where "vertical" is a dictionary of vertical
                measurements for temperature and humidity::

                    data = {
                        "weather": weather_data,
                        "timestamp": bls_time,
                        "vertical": {
                            "temperature": pd.DataFrame,
                            "humidity": pd.DataFrame,
                            }
                        }

        Returns:
            dict: If the input dictionary contained a dictionary under
            the key "vertical" measurements, the dictionary under
            "vertical" is updated with vertical data for water vapour
            pressure, air pressure, mixing ratio, virtual temperature,
            mean sea-level pressure, and potential temperature.
            Otherwise the dictionary is returned unmodified.
        """

        profile = scintillometry.backend.constructions.ProfileConstructor()

        if "vertical" in data:
            data["vertical"] = profile.get_vertical_variables(
                vertical_data=data["vertical"], meteo_data=data["weather"]
            )

        return data

    def match_time_at_threshold(self, series, threshold, lessthan=True):
        """Gets time index of first value in series exceeding threshold.

        Args:
            series (pd.Series): Time series of numeric variable.
            threshold (float): Threshold value.
            lessthan (bool): If True, finds first value less than
                threshold. Otherwise, returns first value greater than
                threshold. Default True.

        Returns:
            pd.Timestamp: Local time of the first value in the series
            that is less than the given threshold.
        """

        if lessthan:
            series_match = series[series.lt(threshold)]
        else:
            series_match = series[series.gt(threshold)]
        if not series_match.empty:
            match_time = series_match.dropna().index[0]
        else:
            match_time = None

        return match_time

    def get_switch_time(self, data, method="sun", local_time=None, ri_crit=0.3):
        """Gets local time of switch between stability conditions.

        To override automatic detection, pass one of the following to
        the <method> argument:

        - **eddy**: eddy covariance (NotImplemented)
        - **sun**: global irradiance (i.e. sunrise)
        - **static**: static stability from potential temperature
          profile
        - **bulk**: bulk Richardson number
        - **lapse**: temperature lapse rate (NotImplemented)
        - **brunt**: Brunt-Väisälä frequency (NotImplemented)

        To manually set the regime switch time, pass a local timestamp
        to <local_time>. This overrides all other methods.

        Args:
            data (dict): Parsed and localised dataframes, containing
                data to construct a potential temperature profile, or
                eddy covariance data, or global irradiance.
            method (str): Method to calculate switch time.
                Default "sun".
            local_time (str): Local time of switch between stability
                conditions. Overrides <method>.
            ri_crit (float): Critical bulk Richardson number for CBL.
                Only used if `method = "bulk"`. For street canyons
                consider values between 0.5 - 1.0 [#zhao2020]_.
                Default 0.25 [#jericevic2006]_.

        Returns:
            str: Local time of switch between stability conditions.

        Raises:
            UnboundLocalError: No data to calculate switch time. Set
                <local_time> manually with `--switch-time`.
            NotImplementedError: Switch time algorithm not implemented
                for <method>.
        """

        if not local_time:
            weather_data = data["weather"]
            if isinstance(method, str):
                method = method.lower()

            if method == "sun" and "global_irradiance" in weather_data.keys():
                print("Using global irradiance to calculate switch time.\n")
                local_time = self.match_time_at_threshold(
                    series=weather_data["global_irradiance"],
                    threshold=20,  # ~sunrise
                    lessthan=False,
                )

            elif "vertical" in data:
                pt_profile = ProfileConstructor()
                if method == "static":  # static stability
                    grad_potential_temperature = pt_profile.get_static_stability(
                        potential_temperature=data["vertical"]["potential_temperature"],
                        scheme="uneven",
                    )
                    heights = grad_potential_temperature.columns[
                        grad_potential_temperature.columns <= 2000
                    ]
                    mean_grad_pot_temperature = grad_potential_temperature[
                        heights
                    ].mean(axis=1, skipna=True)

                    local_time = self.match_time_at_threshold(
                        series=mean_grad_pot_temperature, threshold=0, lessthan=True
                    )
                    data["vertical"][
                        "grad_potential_temperature"
                    ] = grad_potential_temperature

                elif method == "bulk":  # bulk richardson
                    bulk_richardson = pt_profile.get_bulk_richardson(
                        potential_temperature=data["vertical"]["potential_temperature"],
                        meteo_data=weather_data,
                    )
                    local_time = self.match_time_at_threshold(
                        series=bulk_richardson, threshold=ri_crit
                    )
                    data["vertical"]["bulk_richardson"] = bulk_richardson

                elif method == "eddy":  # eddy covariance
                    raise NotImplementedError(
                        f"Switch time algorithm not implemented for '{method}'."
                    )
                else:
                    error_msg = f"Switch time algorithm not implemented for '{method}'."
                    raise NotImplementedError(error_msg)
            if not local_time:
                error_msg = (
                    "No data to calculate switch time.",
                    "Set <local_time> manually with `--switch-time`.",
                )
                raise UnboundLocalError(" ".join(error_msg))
        elif isinstance(local_time, str):
            split_times = local_time.split(sep=":")
            weather_time = pd.Timestamp(
                data["weather"].index[0],
            )
            local_time = weather_time.replace(
                hour=int(split_times[0]), minute=int(split_times[1])
            )

        return local_time

    def plot_switch_time_stability(self, data, local_time, location=""):
        """Plot and save profiles of potential temperature and gradient.

        Args:
            data (dict): Contains dataframes for vertical profiles of
                potential temperature and optionally the gradient of
                potential temperature.
            local_time (pd.Timestamp): Local time of switch between
                stability conditions.
            location (str): Location of data collection. Default empty
                string.

        Returns:
            tuple[plt.Figure, plt.Axes]: Vertical profile of potential
            temperature. If the gradient potential temperature is also
            provided, the two vertical profiles are placed side-by-side
            in separate subplots.
        """

        round_time = self.get_nearest_time_index(
            data=data["potential_temperature"], time_stamp=local_time
        )
        if "grad_potential_temperature" in data:
            fig, ax = scintillometry.visuals.plotting.plot_vertical_comparison(
                dataset=data,
                time_index=round_time,
                keys=["potential_temperature", "grad_potential_temperature"],
                site=location,
            )
        else:
            fig, ax = scintillometry.visuals.plotting.plot_vertical_profile(
                vertical_data=data,
                name="potential_temperature",
                time_idx=round_time,
                site=location,
            )

        mil_time = local_time.strftime("%H%M")
        scintillometry.visuals.plotting.save_figure(
            figure=fig,
            timestamp=data["potential_temperature"].index[0],
            suffix=f"{mil_time}_potential_temperature_profiles",
        )

        return fig, ax

    def calculate_switch_time(
        self, datasets, method="sun", switch_time=None, location=""
    ):
        """Calculates and plots local time of switch in stability.

        Optionally uses vertical measurements in a dictionary under
        `datasets["vertical"]` to plot potential temperature profiles.

        Args:
            datasets (dict): Parsed and localised dataframes, containing
                data to construct a potential temperature profile, or
                eddy covariance data, or global irradiance.
            method (str): Method to calculate switch time.
                Default "sun".
            switch_time (str): Local time of switch between stability
                conditions. Overrides <method>.
            location (str): Location of data collection. Default empty
                string.

        Returns:
            str: Local time of switch between stability conditions.
        """

        switch_time = self.get_switch_time(
            data=datasets, method=method, local_time=switch_time
        )
        if "vertical" in datasets:
            if "potential_temperature" in datasets["vertical"]:
                self.plot_switch_time_stability(
                    data=datasets["vertical"], local_time=switch_time, location=location
                )

        return switch_time

    def iterate_fluxes(self, user_args, z_parameters, datasets, most_id, location=""):
        """Compute sensible heat fluxes with MOST through iteration.

        Trades speed from vectorisation for more accurate convergence.

        Args:
            user_args (argparse.Namespace): Namespace of user arguments.
            z_parameters (dict[float, float]): Tuples of effective and
                mean path height |z_eff| and |z_mean| [m], with
                stability conditions as keys.
            datasets (dict): Contains parsed, tz-aware dataframes, with
                at least |CT2|, wind speed, air density, and
                temperature.
            location (str): Location of data collection. Default empty
                string.

        Returns:
            pd.DataFrame: Interpolated data with additional columns for
            Obukhov length |LOb|, sensible heat flux |H|, friction
            velocity |u*|, and temperature scale |theta*|.
        """

        most_class = scintillometry.backend.iterations.IterationMost()

        interpolated_data = datasets["interpolated"]

        switch_time = self.calculate_switch_time(
            datasets=datasets,
            method=user_args.algorithm,
            switch_time=user_args.switch_time,
            location=location,
        )
        round_time = self.get_nearest_time_index(
            data=interpolated_data, time_stamp=switch_time
        ).strftime("%H:%M")
        iteration_stable = interpolated_data.iloc[
            interpolated_data.index.indexer_between_time("00:00", round_time)
        ].copy(deep=True)
        iteration_stable = most_class.most_method(
            dataframe=iteration_stable,
            eff_h=z_parameters["stable"][0],
            stability="stable",
            coeff_id=most_id,
        )

        iteration_unstable = interpolated_data.iloc[
            interpolated_data.index.indexer_between_time(round_time, "23:59")
        ].copy(deep=True)
        iteration_unstable = most_class.most_method(
            dataframe=iteration_unstable,
            eff_h=z_parameters["unstable"][0],
            stability="unstable",
            coeff_id=most_id,
        )
        iteration_data = pd.concat([iteration_stable, iteration_unstable], sort=True)

        return iteration_data

    def plot_derived_metrics(self, user_args, derived_data, time_id, location=""):
        """Plot and save derived fluxes.

        Args:
            user_args (argparse.Namespace): Namespace of user arguments.
            derived_data (pd.DataFrame): Interpolated tz-aware dataframe
                with column for sensible heat flux under free
                convection.
            time_id (pd.Timestamp): Start time of scintillometer data
                collection.
            location (str): Location of data collection. Default empty
            string.

        Returns:
            plt.Figure: Time series comparing sensible heat fluxes under
            free convection to on-board software.
        """

        figure_convection, _ = scintillometry.visuals.plotting.plot_convection(
            dataframe=derived_data, stability=user_args.regime, site=location
        )
        scintillometry.visuals.plotting.save_figure(
            figure=figure_convection, timestamp=time_id, suffix="free_convection"
        )

        return figure_convection

    def plot_iterated_metrics(self, iterated_data, time_stamp, site_location=""):
        """Plot and save time series and comparison of iterated fluxes.

        Args:
            user_args (argparse.Namespace): Namespace of user arguments.
            derived_data (pd.DataFrame): Interpolated tz-aware dataframe
                with columns for sensible heat fluxes calculated with
                MOST and for free convection.
            time_id (pd.Timestamp): Start time of scintillometer data
                collection.
            site_location (str): Name of scintillometer location.

        Returns:
            tuple[plt.Figure, plt.Figure]: Time series comparing
            sensible heat fluxes under free convection to on-board
            software.
        """

        fig_iter, fig_comp = scintillometry.visuals.plotting.plot_iterated_fluxes(
            iteration_data=iterated_data, time_id=time_stamp, location=site_location
        )

        return fig_iter, fig_comp


class MetricsWorkflow(MetricsFlux, MetricsTopography):
    """Standard workflow for metrics."""

    def __init__(self):
        super().__init__()

    def calculate_standard_metrics(self, arguments, data):
        """Calculates and plots metrics from wrangled data.

        This wrapper function:

        - Calculates effective path heights for all stability
          conditions.
        - Derives |CT2| and sensible heat flux for free convection.
        - Estimates the time where stability conditions change.
        - Calculates sensible heat flux using MOST.
        - Plots time series comparing sensible heat flux for free
          convection |H_free| to on-board software, time series of
          sensible heat flux calculated with MOST |H|, and a comparison
          to sensible heat flux for free convection.
        - Saves plots to disk.

        If this function is imported as a package, mock user arguments
        with an argparse.Namespace object.

        Args:
            arguments (argparse.Namespace): User arguments.
            data (dict): Contains BLS, weather, and transect dataframes,
                an interpolated dataframe at 60s resolution containing
                merged BLS and weather data, a pd.Timestamp object of
                the scintillometer's recorded start time, and optionally
                vertical measurements::

                    data = {
                        "bls": bls_data,
                        "weather": weather_data,
                        "transect": transect_data,
                        "interpolated": interpolated_data,
                        "timestamp": bls_time,
                        "vertical": {
                            "temperature": pd.DataFrame,
                            "humidity": pd.DataFrame,
                            }
                        }

            site (str): Location of data collection. Default empty
                string.

        Returns:
            dict: Input dictionary with additional keys "derivation",
            "iteration" for derived and iterated data, respectively.
        """

        data_timestamp = data["timestamp"]
        z_params = self.get_z_params(user_args=arguments, transect=data["transect"])

        # Compute free convection
        derived_dataframe = self.construct_flux_dataframe(
            user_args=arguments,
            interpolated_data=data["interpolated"],
            z_eff=z_params["None"],
        )

        self.plot_derived_metrics(
            user_args=arguments,
            derived_data=derived_dataframe,
            time_id=data_timestamp,
            location=arguments.location,
        )

        if "vertical" in data:
            data = self.append_vertical_variables(data=data)

        # Compute fluxes through iteration
        iterated_dataframe = self.iterate_fluxes(
            user_args=arguments,
            z_parameters=z_params,
            datasets=data,
            most_id=arguments.most_name,
            location=arguments.location,
        )

        self.plot_iterated_metrics(
            iterated_data=iterated_dataframe,
            time_stamp=data_timestamp,
            site_location=arguments.location,
        )

        data["derivation"] = derived_dataframe
        data["iteration"] = iterated_dataframe

        return data

    def compare_innflux(self, arguments, innflux_data, comparison_data):
        """Compares data to InnFLUX.

        This wrapper function:

        - Plots time series comparing Obukhov lengths and sensible heat
          fluxes between an input dataframe and InnFLUX measurements.
        - Saves plots to disk.

        If this function is imported as a package, mock user arguments
        with an argparse.Namespace object.

        Args:
            arguments (argparse.Namespace): User arguments.
            innflux_data (pd.DataFrame): Eddy covariance data from
                InnFLUX.
            comparison_data (pd.DataFrame): Data to compare to InnFLUX.
            location (str): Location of data collection. Default empty
                string.
        """

        data_timestamp = comparison_data.index[0]
        fig_obu, _ = scintillometry.visuals.plotting.plot_innflux(
            iter_data=comparison_data,
            innflux_data=innflux_data,
            name="obukhov",
            site=arguments.location,
        )
        scintillometry.visuals.plotting.save_figure(
            figure=fig_obu, timestamp=data_timestamp, suffix="innflux_obukhov"
        )
        fig_shf, _ = scintillometry.visuals.plotting.plot_innflux(
            iter_data=comparison_data,
            innflux_data=innflux_data,
            name="shf",
            site=arguments.location,
        )
        scintillometry.visuals.plotting.save_figure(
            figure=fig_shf, timestamp=data_timestamp, suffix="innflux_shf"
        )

        return fig_obu, fig_shf
