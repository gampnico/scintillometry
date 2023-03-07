"""Copyright 2023 Nicolas Gampierakis.

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

import scintillometry.backend.derivations
import scintillometry.backend.iterations
import scintillometry.backend.transects
import scintillometry.visuals.plotting


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

    def iterate_fluxes(self, user_args, z_parameters, interpolated_data, most_id):
        """Compute sensible heat fluxes with MOST through iteration.

        Trades speed from vectorisation for more accurate convergence.

        Args:
            user_args (argparse.Namespace): Namespace of user arguments.
            z_parameters (dict[float, float]): Tuples of effective and
                mean path height |z_eff| and |z_mean| [m], with
                stability conditions as keys.
            interpolated_data (pd.DataFrame): Parsed, tz-aware dataframe
                containing at least |CT2|, wind speed, air density, and
                temperature.

        Returns:
            pd.DataFrame: Interpolated data with additional columns for
            Obukhov length |LOb|, sensible heat flux |H|, friction
            velocity |u*|, and temperature scale |theta*|.
        """

        # Get time where stability conditions change
        most_class = scintillometry.backend.iterations.IterationMost()

        switch_time = most_class.get_switch_time(
            dataframe=interpolated_data, local_time=user_args.switch_time
        )

        iteration_stable = interpolated_data.iloc[
            interpolated_data.index.indexer_between_time("00:00", switch_time)
        ].copy(deep=True)
        iteration_stable = most_class.most_method(
            dataframe=iteration_stable,
            eff_h=z_parameters["stable"][0],
            stability="stable",
            coeff_id=most_id,
        )

        iteration_unstable = interpolated_data.iloc[
            interpolated_data.index.indexer_between_time(switch_time, "23:59")
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
            time_id (pd.TimeStamp): Start time of scintillometer data
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
            time_id (pd.TimeStamp): Start time of scintillometer data
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
            data (dict): Contains BLS, ZAMG, and transect dataframes, an
                interpolated dataframe at 60s resolution containing BLS
                and ZAMG data, and a pd.TimeStamp object of the
                scintillometer's recorded start time::

                    data = {
                        "bls": bls_data,
                        "zamg": zamg_data,
                        "transect": transect_data,
                        "interpolated": interpolated_data,
                        "timestamp": bls_time,
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

        # Compute fluxes through iteration
        iterated_dataframe = self.iterate_fluxes(
            user_args=arguments,
            z_parameters=z_params,
            interpolated_data=data["interpolated"],
            most_id=arguments.most_name,
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
