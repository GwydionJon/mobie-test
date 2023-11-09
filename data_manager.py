from elf.io import open_file
from pathlib import Path
import numpy as np
import trimesh
from collections import defaultdict
from skimage import measure
import pandas as pd
import json
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import logging
from scipy.spatial import ConvexHull


class DataManager:
    def __init__(self, ds_dir_path):
        """_summary_

        :param ds_path: ds_path should be the path to the folder inside `images`that contains the .n5 files.
        :type ds_path: str
        :param organelle_filename: The name of the file that contains the organelle data
        :type organelle_filename: str
        :param scaling_factor: Can be 0,1,2 or 3. The lower the number the higher the resolution, defaults to 3.
        :type scaling_factor: int, optional

        :param original_resolution: The resolution in nm of the raw uncompressed dataset, defaults to (5,5,5)
        :type original_resolution: tuple, optional

        :raises ValueError: Scaling factor must be between 0 and 3.
        """
        logging.basicConfig(level=logging.INFO)

        self.ds_dir_path = Path(ds_dir_path)
        self.paths_dict = self._load_dataset()
        self.resolution = {}
        self.total_volume = {}
        self.datasets = {}
        self.analysis_results = None
        self.mesh_dict = None
        self.collision_dict = None
        self.distance_df = None




    def _load_dataset(self):
        dataset_json = list(self.ds_dir_path.glob("*.json"))[0]
        logging.info(f"Loading dataset from: {str(self.ds_dir_path)}")

        with open(dataset_json, "r") as f:
            dataset_json = json.load(f)
        view_list = list(dataset_json["views"].keys())
        view_list.remove("default")

        paths_dict = {
            filename.split(".n5")[0]: list(self.ds_dir_path.glob(f"**/{filename}.n5"))[
                0
            ]
            for filename in view_list
        }

        logging.info(f"Available segmentations are: {list(paths_dict.keys())}")

        return paths_dict

    def load_organelle_data(
        self,
        organelle_filename,
        organelle_type,
        data_key=None,
        scaling_factor=3,
        original_resolution=(5, 5, 5),
    ):
        if data_key is None:
            data_key = Path(f"setup0/timepoint0/s{scaling_factor}")

        organelle_filename = organelle_filename.split(".n5")[0]
        if organelle_filename not in self.paths_dict.keys():
            raise KeyError(
                f"Organelle {organelle_filename} not found in dataset."
                + f"Available segmentations are: {[key.split('.n5')[0] for key in self.paths_dict.keys()]}"
            )

        complete_path = self.ds_dir_path / self.paths_dict[organelle_filename]

        self.datasets[organelle_type] = self._read_dataset(complete_path, data_key)

        actual_scaling_factor = self.datasets[organelle_type].attrs[
            "downsamplingFactors"
        ]
        self.resolution[organelle_type] = np.asarray(original_resolution) * np.asarray(
            actual_scaling_factor
        )
        self.total_volume[organelle_type] =np.prod(self.resolution[organelle_type] * self.datasets[organelle_type].shape)

        logging.info(f"Successfully read {organelle_filename}.")

    def _read_dataset(self, ds_path, data_key):
        """Read a dataset and select the correct file

        :param ds_path: ds_path should be the path to the folder inside `images`that conatins the .n5 files.
        :type ds_path: str
        :param organelle_filename: The name of the file that contains the organelle data
        :type organelle_filename: str
        :param scaling_factor: Can be 0,1,2 or 3. The lower the number the higher the resolution, defaults to 3.
        :type scaling_factor: int, optional

        """

        logging.debug("Reading dataset from: %s", str(ds_path))

        with open_file(str(ds_path), "r") as f:
            ds = f[data_key]
        return ds

    # for task 1.0
    def get_voxel_properties(self, organelle_types=None, num_pixels_threshhold=10):
        """Calculate the voxel properties for each label in the dataset.
            where applicable this will already account for the spacial resolution.

            Will return the dataframe, but also add it to the class as `self.analysis_results`.
        Properties:
            - Label
            - Volume from voxels
            - Bounding box
            - Bounding box volume
            - Centroid
            - Solidity, hox much of the bounding box is filled.
            - All coordinates, every voxel belonging to this label


        :param num_pixels_threshhold: Ignore objects below this threshold, defaults to 10
        :type num_pixels_threshhold: int, optional

        :return: A DataFrame containing the voxel properties.
        :rtype: pd.DataFrame
        """

        if organelle_types is None:
            organelle_types = self.datasets.keys()
        elif isinstance(organelle_types, str):
            organelle_types = [organelle_types]

        prop_dict = defaultdict(dict)

        for organelle_type in organelle_types:
            properties = measure.regionprops(
                self.datasets[organelle_type], spacing=self.resolution[organelle_type]
            )
            logging.debug("Loading voxel properties:")
            for prop in properties:
                if prop.num_pixels < num_pixels_threshhold:
                    continue
                label = f"{organelle_type}_{prop.label}"
                logging.debug("Processing label: %s", label)
                prop_dict[label]["label"] = f"{organelle_type}_{prop.label}"
                prop_dict[label]["org_type"] = organelle_type

                # as far as i can tell this area attribute is the volume when given a 3d array
                prop_dict[label]["volume_voxels"] = prop.area
                bbox = np.asarray((prop.bbox[:3], prop.bbox[3:]))
                prop_dict[label]["bbox"] = bbox
                prop_dict[label]["bbox_dim_nm"] = bbox[1] * 5 - bbox[0] * 5
                bbox_volume = np.prod(bbox[1] * 5 - bbox[0] * 5)
                prop_dict[label]["bbox_vol_nm"] = bbox_volume

                prop_dict[label]["centroid"] = prop.centroid
                prop_dict[label]["solidity"] = prop.solidity

                prop_dict[label]["all_coords"] = prop.coords

        df = pd.DataFrame(prop_dict).T
        df.index.rename("Label", inplace=True)

        self.analysis_results = df
        return df

    # for task 1.0
    def generate_mesh(self, organelle_types=None):
        """
        Generate a mesh for each label in the dataset.
        Add Surface area and volume to self.analysis_results
        also generates the self.mesh_dict.
        """
        if self.analysis_results is None:
            self.get_voxel_properties()

        if organelle_types is None:
            organelle_types = self.datasets.keys()
        elif isinstance(organelle_types, str):
            organelle_types = [organelle_types]
        mesh_dict = defaultdict(dict)

        for label in self.analysis_results.index:
            organelle_type = label.split("_")[0]
            ds_filtered = self._filter_ds(label, organelle_type)

            try:
                verts, faces, _, _ = measure.marching_cubes(
                    ds_filtered[:], spacing=self.resolution[organelle_type]
                )
            except RuntimeError as e:
                logging.warning("Could not generate mesh for label %s", label)
                logging.warning(np.unique(ds_filtered))
                continue

            # attach the actual mesh to the df (likely not needed)
            mesh_dict[label]["type"] = organelle_type

            mesh_dict[label]["faces"] = faces
            mesh_dict[label]["verts"] = verts

            # calc volume and surface area
            tri_mesh = trimesh.Trimesh(
                vertices=mesh_dict[label]["verts"], faces=mesh_dict[label]["faces"]
            )
            trimesh.repair.fix_inversion(tri_mesh)
            mesh_dict[label]["tri_mesh"] = tri_mesh
            mesh_dict[label]["volume"] = tri_mesh.volume
            mesh_dict[label]["surf_area_mesh_nm"] = tri_mesh.area
            mesh_dict[label]["water_tight"] = tri_mesh.is_watertight
            sphericity_index, flatness_ratio = self._calculate_sphericity_and_flatness(tri_mesh)

            mesh_dict[label]["sphericity"] = sphericity_index
            mesh_dict[label]["flatness_ratio"] = flatness_ratio



        self.analysis_results["surf_area_mesh_nm"] = [
            mesh["surf_area_mesh_nm"] for mesh in mesh_dict.values()
        ]
        self.analysis_results["volume_mesh"] = [
            mesh["volume"] for mesh in mesh_dict.values()
        ]

        # not beeing watertight is a good indicator, that the mesh is located on the border
        self.analysis_results["water_tight"] = [
            mesh["water_tight"] for mesh in mesh_dict.values()
        ]
        # sphericity and flatness ratio
        # sphericity closer to 1 means more spherical
        # flatness ratio closer to 0 means more flat, 1 would be a perfect cube
        self.analysis_results["sphericity"] = [mesh["sphericity"] for mesh in mesh_dict.values()]
        self.analysis_results["flatness_ratio"] = [mesh["flatness_ratio"] for mesh in mesh_dict.values()]



        self.mesh_dict = mesh_dict

    def _calculate_sphericity_and_flatness(self, mesh):
        # Calculate the volume and surface area of the mesh
        volume = mesh.volume
        surface_area = mesh.area

        # Calculate the radius of a sphere with the same volume as the mesh
        sphere_radius = ((3 * volume) / (4 * np.pi))**(1/3)

        # Calculate the surface area of a sphere with the same volume as the mesh
        sphere_surface_area = 4 * np.pi * sphere_radius**2

        # Calculate the sphericity index
        sphericity_index = sphere_surface_area / surface_area

        # flatness
        # Calculate the bounding box of the mesh
        bounding_box = mesh.bounding_box_oriented

        # Calculate the dimensions of the bounding box
        dimensions = bounding_box.extents

        # Calculate the flatness ratio
        flatness_ratio = min(dimensions) / max(dimensions)






        return sphericity_index, flatness_ratio



    # for task 1.1
    def _get_distance(self, obj_id_1, obj_id_2):
        if self.mesh_dict is None:
            self.generate_mesh()

        mesh1 = self.mesh_dict[obj_id_1]["tri_mesh"]
        mesh2 = self.mesh_dict[obj_id_2]["tri_mesh"]

        col_manager_test = trimesh.collision.CollisionManager()
        col_manager_test.add_object(f"mesh{1}", mesh1)
        col_manager_test.add_object(f"mesh{2}", mesh2)

        result = col_manager_test.min_distance_internal()
        return result 
    
    def calculate_distance_matrix(self):
        """Calculate the distance matrix between all objects in the dataset.

        :return: Distance matrix
        :rtype: pd.DataFrame
        """

        if self.analysis_results is None:
            self.generate_mesh()
        num_rows = len(self.analysis_results)
        distance_matrix = np.zeros((num_rows, num_rows))
        for i in range(num_rows):
            for j in range(i, num_rows):
                ind_i = self.analysis_results.index[i]
                ind_j = self.analysis_results.index[j]

                distance = self._get_distance(ind_i, ind_j)
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

        distance_df = pd.DataFrame(
            distance_matrix,
            index=self.analysis_results.index,
            columns=self.analysis_results.index,
        )
        self.distance_df = distance_df
        return distance_df

    # task 1.1
    def _find_neighbour(self, label, max_distance=200, min_distance=0):
        """Find the closest neighbour for a given label.

        :param label: The label to find the closest neighbour for.
        :type label: int
        :param max_distance: The maximum distance between two objects to be considered neighbours in nm, defaults to 200
        :type distance_threshhold: int, optional
        :param min_distance: The minimum distance between two objects to be considered neighbours in nm, defaults to 0
        :type distance_threshhold: int, optional
        :return: The label of the closest neighbour
        :rtype: int
        """

        if self.distance_df is None:
            self.calculate_distance_matrix()

        neighbours = self.distance_df.loc[label]
        neighbours = neighbours[neighbours <= max_distance]
        neighbours = neighbours[neighbours >= min_distance]
        neighbours = neighbours.sort_values()
        if len(neighbours) == 0:
            return None

        return neighbours

    # task 1.1
    def find_neighbours(self, max_distance=200, min_distance=0):
        """Find neighbours in radius for all objects in the dataset.

        :param max_distance: The maximum distance between two objects to be considered neighbours in nm, defaults to 200
        :type distance_threshhold: int, optional
        :param min_distance: The minimum distance between two objects to be considered neighbours in nm, defaults to 0
        :type distance_threshhold: int, optional

        :return: A dictionary containing the neighbours for each label
        :rtype: dict[int, list[int]]
        """
        neighbours_dict = defaultdict(list)
        for label in self.analysis_results.index:
            neighbours = self._find_neighbour(label, max_distance, min_distance)
            if neighbours is None:
                self.analysis_results.loc[label, "num_neighbours"] = 0
                continue
            # this additional loop should deal mith multiple hits at the same distance
            # while also filtering out the self hit
            for neighbours_index in neighbours.index:
                if neighbours_index != label:
                    neighbours_dict[label].append(neighbours_index)

            if label in list(neighbours_dict.keys()):
              self.analysis_results.loc[label, f"num_neighbours_{max_distance}-{min_distance}"] = len(neighbours_dict[label])

        return neighbours_dict

   

    def detect_collisions(self):
        """Detect collisions between all objects in the dataset.

        :return: List of all collision pairs
        :rtype: list[str]
        """
        if self.mesh_dict is None:
            self.generate_mesh()

        col_manager = trimesh.collision.CollisionManager()
        meshes = []
        for mesh_idx, mesh_raw in self.mesh_dict.items():
            mesh = mesh_raw["tri_mesh"]
            col_manager.add_object(mesh_idx, mesh)
            meshes.append(mesh)
        _, names_list, data_list = col_manager.in_collision_internal(
            return_names=True, return_data=True
        )

        # https://trimesh.org/trimesh.boolean.html
        # would be very nice, but seems to require additional installation of blender or openscad
        # https://github.com/mikedh/trimesh/issues/333
        # intersection = trimesh.boolean.intersection(meshes)


        self.collision_dict = defaultdict(lambda: defaultdict(list))
        for data in data_list:
            names = f"{list(data.names)[0]}__{list(data.names)[1]}"
            # add points to al collisions dict
            self.collision_dict[names]["points"].append(data.point)

       

            

        for name,collision in self.collision_dict.items():
            collision_array = np.asarray(collision["points"])
            volume = ConvexHull(collision_array).volume
            self.collision_dict[name]["volume"]=volume

            # add data to collision data dict
            self.collision_dict[name]["Mesh 1"]=name.split("__")[0]
            self.collision_dict[name]["Mesh 2"]=name.split("__")[1]
  


        # add collision data to analysis results
        self.analysis_results["collisions"]=0
        self.analysis_results["total_collisions_volume"]=0.0

        for key, values in self.collision_dict.items():
            for id in key.split("__"):
                self.analysis_results.loc[id, "collisions"] +=1
                self.analysis_results.loc[id, "total_collisions_volume"] += values["volume"]

        self.analysis_results["col_per_area"]= self.analysis_results["collisions"] / self.analysis_results["surf_area_mesh_nm"]
        self.analysis_results["col_per_vol"]= self.analysis_results["collisions"] / self.analysis_results["volume_mesh"]

        return self.analysis_results[["label","surf_area_mesh_nm",	"volume_mesh", "collisions", "total_collisions_volume", "col_per_area", "col_per_vol"]]



        

    def draw_3d_meshes(self, filter_ids=None, exclude_ids=None):
        """Draw the meshes in 3d using plotly, allows for filtering.
            If collisions have been detected they will be drawn as well.


        :param filter_ids: Ids to show in the plot, defaults to None
        :type filter_ids: int or list[int], optional
        :return: The plotly figure
        :rtype: go.Figure
        """
        meshes = []

        if self.mesh_dict is None:
            self.generate_mesh()

        if filter_ids is None:
            index = self.mesh_dict.keys()
        else:
            index = filter_ids

       

        if isinstance(index, str):
            index = [index]
        if isinstance(exclude_ids, str):
            exclude_ids = [exclude_ids]

        if exclude_ids is not None:
            index = [i for i in index if i not in exclude_ids]

        for mesh_idx, mesh_raw in self.mesh_dict.items():
            if mesh_idx not in index:
                continue

            verts = mesh_raw["verts"]
            faces = mesh_raw["faces"]

            # prepare data for plotly
            vertsT = np.transpose(verts)
            facesT = np.transpose(faces)

            go_mesh = go.Mesh3d(
                x=vertsT[0],
                y=vertsT[1],
                z=vertsT[2],
                i=facesT[0],
                j=facesT[1],
                k=facesT[2],
                name=mesh_idx,
                opacity=0.8,            )
            meshes.append(go_mesh)

        if self.collision_dict is not None:
            for key, value in self.collision_dict.items():
                points = value["points"]
                pointsT = np.transpose(points)
                go_points = go.Scatter3d(
                    x=pointsT[0],
                    y=pointsT[1],
                    z=pointsT[2],
                    name="Collision_" + key,
                    mode="markers",
                    marker=dict(
                        size=5,
                        color="red",
                    ),
                    showlegend=True,
                )
                meshes.append(go_points)

        # draw figure
        fig = go.Figure()
        for mesh_ in meshes:
            fig.add_traces(mesh_)

        return fig

    def draw_2d_slices(self, organelle_type, port = 8083):
        """Start a small dash app that allows you to scroll through the slices of the dataset.

        :return: Dash app visualization
        :rtype: Dash
        """

        app = Dash("test")

        app.layout = html.Div(
            [
                html.H4("Cell Stacks"),
                dcc.Graph(id="graph"),
                html.P("Slice:"),
                dcc.Slider(
                    id="slices",
                    min=0,
                    max=self.datasets[organelle_type].shape[2],
                    step=1,
                    value=1,
                ),
            ]
        )

        @app.callback(Output("graph", "figure"), Input("slices", "value"))
        def filter_heatmap(slice):
            ds_slice = self.datasets[organelle_type][
                :, :, slice
            ]  # replace with your own data source
            fig = px.imshow(ds_slice)
            return fig

        app.run_server(debug=True, port=port)

    def _filter_ds(self, filter_index, organelle_type):
        # filter for one value
        # split organell_type from filter value

        filter_value = filter_index.split("_")[1]
        ds_raw = self.datasets[organelle_type][:]
        ds_filtered = np.where(ds_raw == int(filter_value), ds_raw, 0)

        return ds_filtered
