"""Copyright 2023 Scintillometry Contributors.

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

import kneed
import pandas as pd
from sklearn.linear_model import LinearRegression

from scintillometry.backend.constants import AtmosConstants
from scintillometry.backend.constructions import ProfileConstructor
from scintillometry.backend.derivations import DeriveScintillometer
from scintillometry.backend.iterations import IterationMost
from scintillometry.backend.transects import TransectParameters
from scintillometry.visuals.plotting import FigurePlotter
from scintillometry.backend.deprecations import Decorators


class MetricsTopography:
    """Calculate metrics for topographical data.

    Attributes:
        transect (TransectParameters): Inherited methods for calculating
            path heights.
    """

    def __init__(self):
        super().__init__()
        self.transect = TransectParameters()

    def get_path_height_parameters(self, transect, regime=None):
        """Get effective and mean path heights of transect.

        Computes effective and mean path heights of transect under
        stable and unstable conditions, and with no height dependency.
        Prints the effective and mean path height for the user-selected
        stability conditions.

        Select the stability conditions using ``--r, --regime <str>``.

        Args:
            transect (pd.DataFrame): Parsed path transect data.
            regime (str): Target stability condition. Default None.

        Returns:
            dict[str, tuple[np.floating, np.floating]]: Tuples of
            effective and mean path height |z_eff| and |z_mean| [m],
            with stability conditions as keys.
        """

        z_params = self.transect.get_all_path_heights(path_transect=transect)

        self.transect.print_path_heights(
            z_eff=z_params[str(regime)][0],
            z_mean=z_params[str(regime)][1],
            stability=regime,
        )

        return z_params


class MetricsFlux:
    """Calculate metrics for fluxes.

    Attributes:
        constants (AtmosConstants): Inherited atmospheric constants.
        plotting (FigurePlotter): Inherited methods for plotting figures.
        derivation (DeriveScintillometer): Inherited methods for
            deriving parameters and fluxes from scintillometer.
        construction (ProfileConstructor): Inherited methods for
            constructing vertical profiles.
        iteration (IterationMost): Inherited methods for MOST iterative
            method.
    """

    def __init__(self):
        super().__init__()
        self.constants = AtmosConstants()
        self.plotting = FigurePlotter()
        self.derivation = DeriveScintillometer()
        self.construction = ProfileConstructor()
        self.iteration = IterationMost()

    def construct_flux_dataframe(
        self, interpolated_data, z_eff, beam_wavelength=880, beam_error=20
    ):
        """Compute sensible heat flux for free convection.

        Computes sensible heat flux for free convection, plots a
        comparison between the computed flux and the flux recorded by
        the scintillometer, and saves the figure to disk.

        Warning: this will overwrite existing values for |CT2| in
        <interpolated_data>.

        Args:
            interpolated_data (pd.DataFrame): Dataframe containing
                parsed and localised weather and scintillometer data
                with matching temporal resolution.
            z_eff (np.floating): Effective path height, z_eff| [m].
            beam_wavelength (int): Transmitter beam wavelength, nm.
                Default 880 nm.
            beam_error (int): Transmitter beam error, nm. Default 20 nm.

        Keyword Args:
            beam_wavelength (int): Transmitter beam wavelength, nm.
                Default 880 nm.
            beam_error (int): Transmitter beam error, nm. Default 20 nm.

        Returns:
            pd.DataFrame: Interpolated dataframe with additional column
            for sensible heat flux under free convection, and derived
            values for  |CT2| [|K^2m^-2/3|].
        """

        flux_data = self.derivation.compute_fluxes(
            input_data=interpolated_data,
            effective_height=z_eff[0],
            beam_params=(beam_wavelength, beam_error),
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
            Otherwise, the dictionary is returned unmodified.
        """

        if "vertical" in data:
            data["vertical"] = self.construction.get_vertical_variables(
                vertical_data=data["vertical"], meteo_data=data["weather"]
            )

        return data

    def match_time_at_threshold(self, series, threshold, lessthan=True, min_time=None):
        """Gets time index of first value in series exceeding threshold.

        Args:
            series (pd.Series): Time series of numeric variable.
            threshold (float): Threshold value.
            lessthan (bool): If True, finds first value less than
                threshold. Otherwise, returns first value greater than
                threshold. Default True.
            min_time (pd.Timestamp): Time indices below this threshold
                are discarded. Default None.

        Returns:
            pd.Timestamp: Local time of the first value in the series
            that is less than the given threshold.
        """

        if min_time:
            series = series[series.index >= min_time]

        if lessthan:
            series_match = series[series.lt(threshold)]
        else:
            series_match = series[series.gt(threshold)]
        if not series_match.empty:
            match_time = series_match.dropna().index[0]
        else:
            match_time = None

        return match_time

    def get_regression(self, x_data, y_data, intercept=True):
        """Performs regression on labelled data.

        Args:
            x_data (pd.Series): Labelled explanatory data.
            y_data (pd.Series): Labelled response data.
            intercept (bool): If True, calculate intercept (e.g. data is
            not centred). Default True.

        Returns:
            dict: Contains the fitted estimator for regression data, the
            coefficient of determination |R^2|, and predicted values for
            a fitted regression line.
        """

        scatter_frame = pd.merge(
            x_data, y_data, left_index=True, right_index=True, sort=True
        )
        scatter_frame = scatter_frame.dropna(axis=0)
        x_fit_data = scatter_frame.iloc[:, 0].values.reshape(-1, 1)
        y_fit_data = scatter_frame.iloc[:, 1].values.reshape(-1, 1)

        linear_regressor = LinearRegression(fit_intercept=intercept)
        estimator = linear_regressor.fit(x_fit_data, y_fit_data)
        score = estimator.score(x_fit_data, y_fit_data)
        predictions = linear_regressor.predict(x_fit_data)

        regression = {
            "fit": estimator,
            "score": score,
            "regression_line": predictions,
        }

        return regression

    def get_elbow_point(self, series, min_index=None, max_index=None):
        """Calculate elbow point using Kneedle algorithm.

        Only supports convex curves. Noisier curves may have several
        elbow points, in which case the function selects the smallest
        acceptable index.

        Args:
            series (pd.Series): Numeric data following a convex curve.
            min_index (Any): Indices below this threshold are discarded.
                Default None.
            max_index (Any): Indices above this threshold are discarded.
                Default None.

        Returns:
            int: Integer index of elbow point. Returns `None` if no
            elbow point is found.
        """

        if not max_index and not min_index:
            indices = series.index
        else:
            if not max_index:
                max_index = series.index[-1]
            if not min_index:
                min_index = series.index[0]
            indices = series.index[
                (series.index >= min_index) & (series.index <= max_index)
            ]
        if series[indices[-1]] < series[indices[0]]:
            curve_direction = "decreasing"
            online_param = True
        else:
            curve_direction = "increasing"
            online_param = True
        knee = kneed.KneeLocator(
            series[indices],
            indices,
            S=1.5,
            curve="convex",
            online=online_param,
            direction=curve_direction,
            interp_method="interp1d",
        )

        elbows = series[knee.all_elbows_y][
            series[knee.all_elbows_y] < series[indices].mean()
        ]
        if not elbows.empty:
            elbow_index = min(elbows.index)
        elif knee.all_elbows_y:
            elbow_index = min(knee.all_elbows_y)
        else:
            elbow_index = None

        return elbow_index

    def get_boundary_height(self, grad_potential, time_index, max_height=2000):
        """Estimate height of boundary layer from potential temperature.

        Estimates the height of the boundary layer by calculating the
        elbow point of the gradient potential temperature using the
        Kneedle algorithm, i.e. where the gradient potential temperature
        starts to weaken. It is not a substitute for visual inspection.

        Args:
            grad_potential (pd.DataFrame): Vertical measurements,
                gradient potential temperature, |Dtheta/Dz| [|Km^-1|].
            time_index (pd.Timestamp): Local time at which to estimate
                boundary layer height.
            max_height (int): Cutoff height for estimating boundary
                layer height. Default 2000.

        Returns:
            int: Estimated boundary layer height, |z_BL| [m].
        """

        time_index = self.get_nearest_time_index(
            data=grad_potential, time_stamp=time_index
        )
        curve = grad_potential.loc[time_index]
        # set minimum height to 50m to avoid discontinuity errors
        elbow = self.get_elbow_point(series=curve, min_index=50, max_index=max_height)
        if not elbow:
            bl_height = None
        elif elbow >= max_height:
            bl_height = None
        else:
            bl_height = elbow

        if not bl_height:
            print("Failed to estimate boundary layer height.")
        else:
            print(f"Estimated boundary layer height: {bl_height} m.")

        return bl_height

    def compare_lapse_rates(self, air_temperature, saturated, unsaturated):
        """Compares parcel temperatures to find instability.

        Args:
            air_temperature (pd.DataFrame): Vertical measurements,
                temperature, |T| [K].
            saturated (pd.DataFrame): Vertical measurements, saturated
                parcel temperature, |T_sat| [K].
            unsaturated (pd.DataFrame): Vertical measurements,
                unsaturated parcel temperature, |T_unsat| [K].

        Returns:
            tuple[pd.Series[bool], pd.Series[bool]]: Boolean series of
            absolute and conditional instability for heights |z|.
        """

        heights = air_temperature.columns[
            (air_temperature.columns > 0) & (air_temperature.columns <= 2000)
        ]

        absolute_instability = (
            air_temperature[heights].gt(unsaturated[heights]).any(axis=1)
        )
        conditional_instability = (
            air_temperature[heights].lt(saturated[heights]).any(axis=1)
            & air_temperature[heights].gt(saturated[heights]).any(axis=1)
        ) | air_temperature[heights].gt(unsaturated[heights]).any(axis=1)

        return absolute_instability, conditional_instability

    def plot_lapse_rates(
        self, vertical_data, dry_adiabat, local_time, bl_height=None, location=""
    ):
        """Plots comparison of lapse rates and boundary layer height.

        The figures are saved to disk.

        Args:
            vertical_data (dict): Vertical measurements for lapse rates
                and temperatures.
            dry_adiabat (float): Dry adiabatic lapse rate.
            local_time (pd.Timestamp): Local time of switch between
                stability conditions.
            bl_height (float): Boundary layer height, [m]. Default None.
            location (str): Location of data collection. Default empty
                string.

        Returns:
            list[tuple[plt.Figure, plt.Axes]]: Vertical profiles of
            lapse rates on a single axis, and vertical profiles of
            parcel temperatures on a single axis. If a boundary layer
            height is provided, vertical lines denoting its height are
            added to the figures.
        """

        lapse_rates = {
            "environmental_lapse_rate": vertical_data["environmental_lapse_rate"],
            "moist_adiabatic_lapse_rate": vertical_data["moist_adiabatic_lapse_rate"],
        }
        round_time = self.get_nearest_time_index(
            data=next(iter(lapse_rates.values())), time_stamp=local_time
        )
        fig_lapse, axes_lapse = self.plotting.plot_merged_profiles(
            dataset=lapse_rates,
            time_index=round_time,
            vlines={"dry_adiabatic_lapse_rate": dry_adiabat},
            hlines={"boundary_layer_height": bl_height},
            site=location,
            y_lim=2000,
            title="Temperature Lapse Rates",
            x_label=r"Lapse Rate, [Km$^{-1}$]",
        )
        self.plotting.save_figure(
            figure=fig_lapse,
            timestamp=round_time,
            suffix=f"{round_time.strftime('%H%M')}_lapse_rates",
        )

        parcel_temperatures = {
            "temperature": vertical_data["temperature"],
            "unsaturated_temperature": vertical_data["unsaturated_temperature"],
            "saturated_temperature": vertical_data["saturated_temperature"],
        }

        fig_parcel, axes_parcel = self.plotting.plot_merged_profiles(
            dataset=parcel_temperatures,
            time_index=round_time,
            hlines={"boundary_layer_height": bl_height},
            site=location,
            y_lim=2000,
            title="Vertical Profiles of Parcel Temperature",
            x_label="Temperature, [K]",
        )
        self.plotting.save_figure(
            figure=fig_parcel,
            timestamp=round_time,
            suffix=f"{round_time.strftime('%H%M')}_parcel_temperatures",
        )

        return [(fig_lapse, axes_lapse), (fig_parcel, axes_parcel)]

    def get_switch_time_vertical(self, data, method="static", ri_crit=0.25):
        """Gets local time of switch between stability conditions.

        Pass one of the following to the <method> argument:

        - **eddy**: eddy covariance (NotImplemented)
        - **static**: static stability from potential temperature
          profile
        - **bulk**: bulk Richardson number
        - **lapse**: temperature lapse rate
        - **brunt**: Brunt-Väisälä frequency (NotImplemented)

        Args:
            data (dict): Parsed and localised dataframes, containing
                vertical measurements to construct a potential
                temperature profile or calculate lapse rates.
            method (str): Method to calculate switch time.
                Default "static".
            ri_crit (float): Critical bulk Richardson number for CBL.
                Only used if `method = "bulk"`. For street canyons
                consider values between 0.5 - 1.0 [#zhao2020]_.
                Default 0.25 [#jericevic2006]_.

        Returns:
            pd.Timestamp: Local time of switch between stability
            conditions.

        Raises:
            NotImplementedError: Switch time algorithm not implemented
                for <method>.
        """

        # static stability
        if method == "static" and "grad_potential_temperature" in data["vertical"]:
            heights = data["vertical"]["grad_potential_temperature"].columns[
                data["vertical"]["grad_potential_temperature"].columns <= 2000
            ]
            negative_grad = (
                data["vertical"]["grad_potential_temperature"][heights]
                .lt(0)
                .any(axis=1)
            )
            local_time = self.match_time_at_threshold(
                series=negative_grad,  # since True == 1
                threshold=0,
                lessthan=False,
                min_time=data["timestamp"],
            )

        elif method == "bulk":  # bulk richardson
            bulk_richardson = self.construction.get_bulk_richardson(
                potential_temperature=data["vertical"]["potential_temperature"],
                meteo_data=data["weather"],
            )
            local_time = self.match_time_at_threshold(
                series=bulk_richardson, threshold=ri_crit, lessthan=True
            )
            data["vertical"]["bulk_richardson"] = bulk_richardson

        elif method == "lapse":  # lapse rates
            _, conditional_instability = self.compare_lapse_rates(
                air_temperature=data["vertical"]["temperature"],
                saturated=data["vertical"]["saturated_temperature"].dropna(),
                unsaturated=data["vertical"]["unsaturated_temperature"].dropna(),
            )
            local_time = self.match_time_at_threshold(
                series=conditional_instability,
                threshold=0,
                lessthan=False,
                min_time=data["timestamp"],
            )

        else:
            error_msg = f"Switch time algorithm not implemented for '{method}'."
            raise NotImplementedError(error_msg)

        return local_time

    def get_switch_time(self, data, method="sun", local_time=None, ri_crit=0.25):
        """Gets local time of switch between stability conditions.

        To override automatic detection, pass one of the following to
        the <method> argument:

        - **eddy**: eddy covariance (NotImplemented)
        - **sun**: global irradiance (i.e. sunrise)
        - **static**: static stability from potential temperature
          profile
        - **bulk**: bulk Richardson number
        - **lapse**: temperature lapse rate
        - **brunt**: Brunt-Väisälä frequency (NotImplemented)

        To manually set the regime switch time, pass a localised
        timestamp or string to <local_time>. This overrides all other
        methods.

        Args:
            data (dict): Parsed and localised dataframes, containing
                data to construct a potential temperature profile, or
                eddy covariance data, or global irradiance.
            method (str): Method to calculate switch time.
                Default "sun".
            local_time (Union[str, pd.Timestamp]): Local time of switch
                between stability conditions. Overrides <method>.
                Default None.
            ri_crit (float): Critical bulk Richardson number for CBL.
                Only used if `method = "bulk"`. For street canyons
                consider values between 0.5 - 1.0 [#zhao2020]_.
                Default 0.25 [#jericevic2006]_.

        Returns:
            pd.Timestamp: Local time of switch between stability
            conditions.

        Raises:
            UnboundLocalError: No data to calculate switch time. Set
                <local_time> manually with `--switch-time`.
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
                local_time = self.get_switch_time_vertical(
                    data=data, method=method, ri_crit=ri_crit
                )
            if not local_time:
                error_msg = (
                    "No data to calculate switch time.",
                    "Set <local_time> manually with `--switch-time`.",
                )
                raise UnboundLocalError(" ".join(error_msg))
        elif isinstance(local_time, str):
            split_times = local_time.split(sep=":")
            start_time = data["timestamp"]
            local_time = start_time.replace(
                hour=int(split_times[0]), minute=int(split_times[1])
            )

        print(f"Stability conditions change at: {local_time.strftime('%H:%M %Z')}")

        return local_time

    def plot_switch_time_stability(self, data, local_time, location="", bl_height=None):
        """Plot and save profiles of potential temperature and gradient.

        Args:
            data (dict): Contains dataframes for vertical profiles of
                potential temperature and optionally the gradient of
                potential temperature.
            local_time (pd.Timestamp): Local time of switch between
                stability conditions.
            location (str): Location of data collection. Default empty
                string.
            bl_height (int): Boundary layer height. Default None.

        Returns:
            list[tuple[plt.Figure, plt.Axes]]: Vertical profile of
            potential temperature. If the gradient potential temperature
            is also provided, the two vertical profiles are placed
            side-by-side in separate subplots.
        """

        round_time = self.get_nearest_time_index(
            data=data["potential_temperature"], time_stamp=local_time
        )
        mil_time = round_time.strftime("%H%M")

        fig, ax = self.plotting.plot_vertical_profile(
            vertical_data=data,
            name="potential_temperature",
            time_idx=round_time,
            site=location,
            y_lim=2000,
            hlines={"boundary_layer_height": bl_height},
        )
        self.plotting.save_figure(
            figure=fig,
            timestamp=local_time,
            suffix=f"{mil_time}_potential_temperature_2km",
        )

        if "grad_potential_temperature" in data:
            fig, ax = self.plotting.plot_vertical_comparison(
                dataset=data,
                time_index=round_time,
                keys=["potential_temperature", "grad_potential_temperature"],
                site=location,
                hlines={"boundary_layer_height": bl_height},
            )
            self.plotting.save_figure(
                figure=fig,
                timestamp=local_time,
                suffix=f"{mil_time}_potential_temperature_profiles",
            )

            fig_grad, _ = self.plotting.plot_vertical_profile(
                vertical_data=data,
                name="grad_potential_temperature",
                time_idx=round_time,
                site=location,
                y_lim=2000,
                hlines={"boundary_layer_height": bl_height},
            )
            self.plotting.save_figure(
                figure=fig_grad,
                timestamp=local_time,
                suffix=f"{mil_time}_gradient_potential_temperature_2km",
            )

        return [(fig, ax)]

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
            switch_time (Union[str, pd.Timestamp]): Local time of switch
                between stability conditions. Overrides <method>.
                Default None.
            location (str): Location of data collection. Default empty
                string.

        Returns:
            pd.Timestamp: Local time of switch between stability
            conditions.
        """

        switch_time = self.get_switch_time(
            data=datasets, method=method, local_time=switch_time
        )
        if "vertical" in datasets:
            vertical_data = datasets["vertical"]
            if "grad_potential_temperature" in vertical_data:
                layer_height = self.get_boundary_height(
                    grad_potential=vertical_data["grad_potential_temperature"],
                    time_index=switch_time,
                    max_height=2000,
                )
                self.plot_switch_time_stability(
                    data=vertical_data,
                    local_time=switch_time,
                    location=location,
                    bl_height=layer_height,
                )
                if "environmental_lapse_rate" in vertical_data:
                    self.plot_lapse_rates(
                        vertical_data=datasets["vertical"],
                        dry_adiabat=self.constants.dalr,
                        bl_height=layer_height,
                        local_time=switch_time,
                        location=location,
                    )

        return switch_time

    def iterate_fluxes(
        self,
        z_parameters,
        datasets,
        most_id="an1988",
        algorithm="sun",
        switch_time=None,
        location="",
    ):
        """Compute sensible heat fluxes with MOST through iteration.

        Trades speed from vectorisation for more accurate convergence.

        Args:
            z_parameters (dict[str, tuple[float, float]): Tuples of
                effective and mean path height |z_eff| and |z_mean| [m],
                with stability conditions as keys.
            datasets (dict): Contains parsed, tz-aware dataframes, with
                at least |CT2|, wind speed, air density, and
                temperature.
            most_id (str): MOST coefficients for unstable and stable
                conditions. Default "an1988".
            algorithm (str): Method to calculate switch time.
                Default "sun".
            switch_time (Union[str, pd.Timestamp]): Local time of switch
                between stability conditions. Overrides <method>.
                Default None.
            location (str): Location of data collection. Default empty
                string.

        Returns:
            pd.DataFrame: Interpolated data with additional columns for
            Obukhov length |LOb|, sensible heat flux |H|, friction
            velocity |u*|, and temperature scale |theta*|.
        """

        interpolated_data = datasets["interpolated"]

        switch_time = self.calculate_switch_time(
            datasets=datasets,
            method=algorithm,
            switch_time=switch_time,
            location=location,
        )
        round_time = self.get_nearest_time_index(
            data=interpolated_data, time_stamp=switch_time
        ).strftime("%H:%M")
        iteration_stable = interpolated_data.iloc[
            interpolated_data.index.indexer_between_time("00:00", round_time)
        ].copy(deep=True)
        iteration_stable = self.iteration.most_method(
            dataframe=iteration_stable,
            eff_h=z_parameters["stable"][0],
            stability="stable",
            coeff_id=most_id,
        )

        iteration_unstable = interpolated_data.iloc[
            interpolated_data.index.indexer_between_time(round_time, "23:59")
        ].copy(deep=True)
        iteration_unstable = self.iteration.most_method(
            dataframe=iteration_unstable,
            eff_h=z_parameters["unstable"][0],
            stability="unstable",
            coeff_id=most_id,
        )
        iteration_data = pd.concat([iteration_stable, iteration_unstable], sort=True)

        return iteration_data

    def plot_derived_metrics(self, derived_data, time_id, regime=None, location=""):
        """Plot and save derived fluxes.

        Args:
            derived_data (pd.DataFrame): Interpolated tz-aware dataframe
                with column for sensible heat flux under free
                convection.
            time_id (pd.Timestamp): Start time of scintillometer data
                collection.
            regime (str): Stability condition. Default None.
            location (str): Location of data collection. Default empty
            string.

        Returns:
            list[tuple[plt.Figure, plt.Axes]]: Time series comparing
            sensible heat fluxes under free convection to on-board
            software.
        """

        fig_convection, ax_convection = self.plotting.plot_convection(
            dataframe=derived_data, stability=regime, site=location
        )
        self.plotting.save_figure(
            figure=fig_convection, timestamp=time_id, suffix="free_convection"
        )
        derived_plots = [(fig_convection, ax_convection)]

        return derived_plots

    @Decorators.deprecated_argument(
        stage="pending", version="1.0.5", site_location="location"
    )
    def plot_iterated_metrics(self, iterated_data, time_stamp, location=""):
        """Plots and saves iterated SHF, comparison to free convection.

        .. todo::
            ST-126: Deprecate the argument `site_location` for
                `location`.

        Args:
            iterated_data (pd.DataFrame): TZ-aware with columns for
                sensible heat fluxes calculated for free convection
                |H_free|, and by MOST |H|.
            time_stamp (pd.Timestamp): Local time of data collection.
            location (str): Location of data collection. Default empty
                string.

        Returns:
            list[tuple[plt.Figure, plt.Axes]]: Time series of sensible
            heat flux calculated through MOST, and a comparison to
            sensible heat flux under free convection.
        """

        shf_plot = self.plotting.plot_generic(iterated_data, "shf", site=location)
        self.plotting.save_figure(
            figure=shf_plot[0], timestamp=time_stamp, suffix="shf"
        )

        comparison_plot = self.plotting.plot_comparison(
            df_01=iterated_data,
            df_02=iterated_data,
            keys=["H_free", "shf"],
            labels=["Free Convection", "Iteration"],
            site=location,
        )
        self.plotting.save_figure(
            figure=comparison_plot[0], timestamp=time_stamp, suffix="shf_comp"
        )

        return [shf_plot, comparison_plot]


class MetricsWorkflow(MetricsFlux, MetricsTopography):
    """Standard workflow for metrics."""

    def __init__(self):
        super().__init__()

    def calculate_standard_metrics(
        self,
        data,
        regime=None,
        most_name="an1988",
        method="sun",
        switch_time=None,
        location="",
        beam_wavelength=880,
        beam_error=20,
        **kwargs,
    ):
        """Calculates and plots metrics from wrangled data.

        This wrapper function:

        - Calculates effective path heights for all stability
          conditions.
        - Derives |CT2| and sensible heat flux for free convection.
        - Estimates the time when stability conditions change.
        - Calculates sensible heat flux using MOST.
        - Plots time series comparing sensible heat flux for free
          convection |H_free| to on-board software, time series of
          sensible heat flux calculated with MOST |H|, and a comparison
          to sensible heat flux for free convection.
        - Saves plots to disk.

        If this function is imported as a package, mock user arguments
        with an argparse.Namespace object.

        Args:
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

            regime (str): Target stability condition. Default None.
            most_name (str): MOST coefficients for unstable and stable
                conditions. Default "an1988".
            method (str): Method to calculate switch time.
                Default "sun".
            switch_time (Union[str, pd.Timestamp]): Local time of switch
                between stability conditions. Overrides <method>.
                Default None.
            location (str): Location of data collection. Default empty
                string.
            beam_wavelength (int): Transmitter beam wavelength, nm.
                Default 880 nm.
            beam_error (int): Transmitter beam error, nm. Default 20 nm.

        Returns:
            dict: Input dictionary with additional keys "derivation",
            "iteration" for derived and iterated data, respectively.
        """

        data_timestamp = data["timestamp"]
        z_params = self.get_path_height_parameters(
            transect=data["transect"], regime=regime
        )

        # Compute free convection
        derived_dataframe = self.construct_flux_dataframe(
            interpolated_data=data["interpolated"],
            z_eff=z_params["None"],
            beam_wavelength=beam_wavelength,
            beam_error=beam_error,
        )

        self.plot_derived_metrics(
            derived_data=derived_dataframe,
            time_id=data_timestamp,
            location=location,
        )

        if "vertical" in data:
            data = self.append_vertical_variables(data=data)

        # Compute fluxes through iteration
        iterated_dataframe = self.iterate_fluxes(
            z_parameters=z_params,
            datasets=data,
            most_id=most_name,
            algorithm=method,
            switch_time=switch_time,
            location=location,
        )

        self.plot_iterated_metrics(
            iterated_data=iterated_dataframe,
            time_stamp=data_timestamp,
            location=location,
        )

        data["derivation"] = derived_dataframe
        data["iteration"] = iterated_dataframe

        return data

    def compare_innflux(self, own_data, innflux_data, location=""):
        """Compares SHF and Obukhov lengths to innFLUX measurements.

        This wrapper function:

        - Plots time series comparing Obukhov lengths and sensible heat
          fluxes between an input dataframe and innFLUX measurements.
        - Saves plots to disk.

        If this function is imported as a package, mock user arguments
        with an argparse.Namespace object.

        Args:
            own_data (pd.DataFrame): Labelled data for SHF and Obukhov
                length.
            innflux_data (pd.DataFrame): Eddy covariance data from
                innFLUX.
            location (str): Location of data collection. Default empty
                string.

        Returns:
            list[tuple[plt.Figure, plt.Axes]]: Time series comparing
            Obukhov length and sensible heat flux to innFLUX
            measurements.
        """

        data_timestamp = own_data.index[0]
        obukhov_plot = self.plotting.plot_innflux(
            iter_data=own_data,
            innflux_data=innflux_data,
            name="obukhov",
            site=location,
        )
        self.plotting.save_figure(
            figure=obukhov_plot[0], timestamp=data_timestamp, suffix="innflux_obukhov"
        )
        obukhov_regression = self.get_regression(
            x_data=own_data["obukhov"], y_data=innflux_data["obukhov"], intercept=True
        )
        obukhov_regression_plot = self.plotting.plot_scatter(
            x_data=own_data["obukhov"],
            y_data=innflux_data["obukhov"],
            sources=["MOST Iteration", "innFLUX"],
            name="obukhov",
            score=obukhov_regression["score"],
            regression_line=obukhov_regression["regression_line"],
            site=location,
        )
        self.plotting.save_figure(
            figure=obukhov_regression_plot[0],
            timestamp=data_timestamp,
            suffix="innflux_obukhov_regression",
        )

        shf_plot = self.plotting.plot_innflux(
            iter_data=own_data,
            innflux_data=innflux_data,
            name="shf",
            site=location,
        )
        self.plotting.save_figure(
            figure=shf_plot[0], timestamp=data_timestamp, suffix="innflux_shf"
        )
        shf_regression = self.get_regression(
            x_data=own_data["obukhov"], y_data=innflux_data["obukhov"], intercept=True
        )
        shf_regression_plot = self.plotting.plot_scatter(
            x_data=own_data["shf"],
            y_data=innflux_data["shf"],
            sources=["MOST Iteration", "innFLUX"],
            name="shf",
            score=shf_regression["score"],
            regression_line=shf_regression["regression_line"],
            site=location,
        )
        self.plotting.save_figure(
            figure=shf_regression_plot[0],
            timestamp=data_timestamp,
            suffix="innflux_shf_regression",
        )

        plots = [obukhov_plot, shf_plot, obukhov_regression_plot, shf_regression_plot]

        return plots

    def compare_eddy(self, own_data, ext_data, source="innflux", location=""):
        """Compares data to an external source of eddy covariance data.

        Plots and saves time series comparing Obukhov lengths and
        sensible heat fluxes between an input dataframe and external
        eddy covariance measurements.

        If this function is imported as a package, mock user arguments
        with an argparse.Namespace object.

        Args:
            own_data (pd.DataFrame): Labelled data.
            ext_data (pd.DataFrame): Eddy covariance data from an
                external source.
            source (str): Data source of vertical measurements.
                Currently supports processed innFLUX data.
                Default "innflux".
            location (str): Location of data collection. Default empty
                string.

        Returns:
            list[tuple[plt.Figure, plt.Axes]]: Time series comparing
            Obukhov length and sensible heat flux to innFLUX
            measurements.

        Raises:
            NotImplementedError: <source> measurements are not
                supported. Use "innflux".

        """

        if source.lower() == "innflux":
            eddy_plots = self.compare_innflux(
                own_data=own_data, innflux_data=ext_data, location=location
            )
        else:
            error_msg = (
                f"{source.title()} measurements are not supported. Use 'innflux'."
            )
            raise NotImplementedError(error_msg)

        return eddy_plots
