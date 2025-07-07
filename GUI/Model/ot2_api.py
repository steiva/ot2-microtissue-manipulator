import requests
import json

class OpentronsAPI():

    def __init__(self) -> None:
        self.ROBOT_IP = "169.254.241.245"
        self.HEADERS = {"opentrons-version": "3"}
        self.PIPETTE = "p300_single_gen2"
        self.runs_url = f"http://{self.ROBOT_IP}:31950/runs"
        self.lights_url = f"http://{self.ROBOT_IP}:31950/robot/lights"
        self.home_url = f"http://{self.ROBOT_IP}:31950/robot/home"
        self.protocols_url = f"http://{self.ROBOT_IP}:31950/protocols"
        self.run_id = None
        self.pipette_id = None
        self.commands_url = None # None because commands are associated with a run which needs to be initiated first.
        self.protocol_id = None
        self.labware_dct = {str(i): None for i in range(1, 12)}

    def post(self, url: str, headers: dict, params: dict = None, data: dict = None, files: list = None) -> requests.models.Response:
        """Post method to post to HTTP API.

        Args:
            url (str): Endpoint URL.
            headers (dict): Headers.
            params (dict, optional): Optional parameters, like 'waitUntilComplete'. Defaults to None.
            data (dict, optional): Usually a dictionary with a command. Defaults to None.
            files (list, optional): A list of file paths that need to be uploaded to the robot's onboard RaspberryPi. Defaults to None.

        Returns:
            requests.models.Response: a responce from the robot's server.
        """        
        r = requests.post(
            url=url,
            headers=headers,
            params=params,
            data=data,
            files=files)
        return r
    
    def get(self, url: str, headers: dict) -> requests.models.Response:
        """Get method to query the HTTP API.

        Args:
            url (str): Endpoint URL.
            headers (dict): Headers.

        Returns:
            requests.models.Response: a responce from the robot's server.
        """        
        r = requests.get(
            url=url,
            headers=headers)
        return r
    
    def display_responce(self, responce: requests.models.Response) -> None:
        """Simple method to print the responce from the server. The responce is formatted to be more readable.

        Args:
            responce (requests.models.Response): a responce object from the robot's server.
        """        
        json_formatted_str = json.dumps(json.loads(responce.text), indent = 2)
        print(f"Request status:\n{responce}\n{json_formatted_str}")
    
    def toggle_lights(self, verbose: bool = False) -> None:
        """Method to toggle the OT-2's top LED strips on/off.

        Args:
            verbose (bool, optional): Print the responce from server or not. Defaults to False.
        """        
        current_status = self.get(self.lights_url, self.HEADERS)
        current_status = json.loads(current_status.text)
        is_on = current_status['on']
        responce_data = json.dumps({"on": not(is_on)})
        toggle_responce = self.post(url = self.lights_url, headers=self.HEADERS, data=responce_data)
        if verbose:
            self.display_responce(toggle_responce)

    def create_run(self, protocol_id: str = None, verbose: bool = True) -> None:
        """Method to create a run for the OT-2. A run needs to be created before any commands 
           like movement, aspiration, etc. are executed, but not for basic commands such as 
           light toggle or homing the robot.  

        Args:
            protocol_id (str, optional): Optional argument to include protocol that is being executed. Defaults to None.
            verbose (bool, optional): Print the responce from server or not. Defaults to True.
        """        
        if protocol_id:
            protocol_id_payload = json.dumps({"data":{"protocolId": protocol_id}})
            r = self.post(self.runs_url, self.HEADERS, data = protocol_id_payload)
        else:
            r = self.post(self.runs_url, self.HEADERS)
        run_id = json.loads(r.text)['data']['id']
        self.run_id = run_id
        if verbose:
            self.display_responce(r)

    def home_robot(self, verbose: bool = True) -> None:
        """Method to home the robot's gantry. Needs to be performed at least once before operation. 

        Args:
            verbose (bool, optional): Print the responce from server or not. Defaults to True.
        """        
        command_dict = {"target": "robot"}
        command_payload = json.dumps(command_dict)
        r = self.post(url = self.home_url, headers=self.HEADERS, data = command_payload)
        if verbose:
            self.display_responce(r)

    def load_pipette(self, mount: str = 'left', verbose: bool = True) -> None:
        """Method that provides the currently attached pipette with a unique ID. Needed to keep track of
           attached/detached pipette tips, etc.

        Args:
            mount (str, optional): 'left' or 'right'. Defaults to 'left'.
            verbose (bool, optional): Print the responce from server or not. Defaults to True.
        """        
        if self.run_id is None:
            print('No current run associated with the robot. Create a run first.')
            return
        self.commands_url = f"{self.runs_url}/{self.run_id}/commands"
        command_dict = {
            "data": {
                "commandType": "loadPipette",
                "params": {
                    "pipetteName": self.PIPETTE,
                    "mount": mount
                },
                "intent": "setup"
            }
        }
        command_payload = json.dumps(command_dict)
        r = self.post(url = self.commands_url, headers= self.HEADERS,
                      params={"waitUntilComplete": True}, data = command_payload)
        if json.loads(r.text)['data']['status'] == 'succeeded':
            self.pipette_id = json.loads(r.text)['data']['result']['pipetteId']
        
        if verbose:
            self.display_responce(r)

    def upload_protocol(self, PROTOCOL_FILE: str, LABWARE_FILE: str = None, verbose: bool = True) -> None:
        """Method that allows to upload a protocol file (as well as custom labware) to the onboard
           RaspberryPi of the robot. The custom labware has to be uploaded together with a protocol file.

        Args:
            PROTOCOL_FILE (str): path to protocol file in your file system.
            LABWARE_FILE (str, optional): path to a custom labware file. Defaults to None.
            verbose (bool, optional): Print the responce from server or not. Defaults to True.
        """        
        protocol_file_payload = open(PROTOCOL_FILE, "rb")
        files = [("files", protocol_file_payload)]
        if LABWARE_FILE:
            labware_file_payload = open(LABWARE_FILE, "rb")
            files.append(("files", labware_file_payload))

        r = self.post(url = self.protocols_url, headers = self.HEADERS, files = files)

        if verbose:
            self.display_responce(r)
        r_dict = json.loads(r.text)
        self.protocol_id = r_dict["data"]["id"]
        print(f"Protocol ID:\n{self.protocol_id}")

        protocol_file_payload.close()
        if LABWARE_FILE:
            labware_file_payload.close()

    def run_protocol(self) -> None:
        """Method for execution of a run labeled as the current one. 
           TODO: reconsider if a run_id should be fed in as a argument instead.
        """        
        if self.run_id is None:
            print('No current run associated with the robot. Create a run with a protocol attached first.')
            return
        actions_url = f"{self.runs_url}/{self.run_id}/actions"
        action_payload = json.dumps({"data":{"actionType": "play"}})

        r = requests.post(
            url=actions_url,
            headers=self.HEADERS,
            data=action_payload
	)
        
    def move_to_coordinates(self, coordinates: tuple, min_z_height: float = 20.0, force_direct: bool = False, verbose: bool = True) -> None:
        """Method to move the robot's end-effector (mount tip or pipette tip if it is attached) to a coordinate position.

        Args:
            coordinates (tuple): Tuple of x,y,z coordinates. 
            min_z_height (float, optional): Minimum height above the platform. Defaults to 20.0.
            force_direct (bool, optional): Force direct movement from one point to the other. Defaults to False.
            verbose (bool, optional): Print the responce from server or not. Defaults to True.
        """        
        if self.pipette_id is None:
            print('Pipette not loaded. Load pipette first.')
            return

        if len(coordinates) != 3:
            print(f'Coordinate tuple needs 3 values, got {len(coordinates)} instead.')
            return

        x,y,z = coordinates
        command_dict = {
            "data": {
                "commandType": "moveToCoordinates",
                "params": {
                    "coordinates": {"x": x, "y": y, "z": z},
                    "minimumZHeight": min_z_height,
                    "forceDirect": force_direct,
                    "pipetteId": self.pipette_id
                },
                "intent": "setup"
            }
        }

        command_payload = json.dumps(command_dict)
        r = self.post(url = self.commands_url, headers = self.HEADERS,
                      params={"waitUntilComplete": True}, data = command_payload)
        if verbose == True:
            self.display_responce(r)

    def move_relative(self, axis: str, distance: float, verbose: bool = False) -> None:
        """Method to move the robot's end-effector (mount tip or pipette tip if it is attached) a relative distance along an axis.

        Args:
            axis (str): Should be 'x', 'y' or 'z'.
            distance (float): Distance to move along the axis in mm.
            verbose (bool, optional): Print the responce from server or not. Defaults to False.
        """        
        if self.pipette_id is None:
            print('Pipette not loaded. Load pipette first.')
            return
        
        if axis not in ('x', 'y', 'z'):
            print('Axis argument must be either x, y or z ...')
            return
        
        command_dict = {
            "data": {
                "commandType": "moveRelative",
                "params": {
                    "axis": axis,
                    "distance": distance,
                    "pipetteId": self.pipette_id
                },
                "intent": "setup"
            }
        }

        command_payload = json.dumps(command_dict)
        r = self.post(url = self.commands_url, headers = self.HEADERS,
                      params={"waitUntilComplete": True}, data = command_payload)
        if verbose == True:
            self.display_responce(r)

    def get_position(self, verbose: bool = True) -> dict:
        """Get current position (mount tip or pipette tip if attached). Method is a bit hacky, as it rely's
           on the savePosition command which is used for calibration.

        Args:
            verbose (bool, optional): Print the responce from server or not. Defaults to True.

        Returns:
            dict: returns dictionary with coordinates: {"x":, "y":, "z":}.
        """        
        if self.pipette_id is None or self.commands_url is None:
            print('Pipette not loaded. Load pipette first.')
            return
        
        command_dict = {
            "data" : {
                "commandType": "savePosition",
                "params": {
                    "pipetteId": self.pipette_id,
                },
                "intent": "setup"
            }
        }

        command_payload = json.dumps(command_dict)
        r = self.post(url = self.commands_url, headers = self.HEADERS, 
                  params={"waitUntilComplete": True}, data = command_payload)
        
        r_dict = json.loads(r.text)
        coordinates = r_dict['data']['result']['position']
        if verbose:
            print(coordinates)
        return coordinates
    
    def load_labware(self, TIP_RACK: str, slot_name: int, verbose: bool = True) -> None:

        if self.commands_url is None:
            print('Pipette not loaded. Load pipette first.')
            return
        
        command_dict = {
            "data": {
                "commandType": "loadLabware",
                "params": {
                    "location": {"slotName": str(slot_name)},
                    "loadName": TIP_RACK,
                    "namespace": "opentrons",
                    "version": 1
                },
                "intent": "setup"
            }
        }

        command_payload = json.dumps(command_dict)
        r = self.post(url = self.commands_url, headers = self.HEADERS,
                  params={"waitUntilComplete": True}, data = command_payload)
        
        r_dict = json.loads(r.text)
        labware_id = r_dict["data"]["result"]["labwareId"]
        self.labware_dct[str(slot_name)] = labware_id
        if verbose == True:
            print(f"Labware ID:\n{labware_id}\n")

    def pick_up_tip(self, labware_id: str, well_name: str, xyz_offset: tuple = (0,0,0), verbose: bool = False) -> None:
        """Method to pick up a tip from a well in a tip rack.

        Args:
            labware_id (str): unique ID of the labware produced in the load_labware method.
            well_name (str): coordinate of the tip, e.g. 'A1'.
            xyz_offset (tuple, optional): xyz offset. Defaults to (0,0,0).
            verbose (bool, optional): Print the responce from server or not. Defaults to False.
        """        

        if self.pipette_id is None or self.commands_url is None:
            print('Pipette not loaded. Load pipette first.')
            return
        
        command_dict = {
            "data": {
                "commandType": "pickUpTip",
                "params": {
                    "labwareId": labware_id,
                    "wellName": well_name,
                    "wellLocation": {
                        "origin": "top", 
                        "offset": {"x": xyz_offset[0], "y": xyz_offset[1], "z": xyz_offset[2]}
                    },
                    "pipetteId": self.pipette_id
                },
                "intent": "setup"
            }
        }

        command_payload = json.dumps(command_dict)
        r = self.post(url = self.commands_url, headers = self.HEADERS,
                  params={"waitUntilComplete": True}, data = command_payload)
        
        if verbose == True:
            self.display_responce(r)

    def drop_tip(self, labware_id: str, well_name: str, xyz_offset: tuple = (0,0,0), verbose: bool = False) -> None:

        if self.pipette_id is None or self.commands_url is None:
            print('Pipette not loaded. Load pipette first.')
            return
        
        command_dict = {
            "data": {
                "commandType": "dropTip",
                "params": {
                    "labwareId": labware_id,
                    "wellName": well_name,
                    "wellLocation": {
                        "origin": "top",
                        "offset": {"x": xyz_offset[0], "y": xyz_offset[1], "z": xyz_offset[2]}
                    },
                    "pipetteId": self.pipette_id
                },
                "intent": "setup"
            }
        }

        command_payload = json.dumps(command_dict)
        r = self.post(url = self.commands_url, headers = self.HEADERS,
                  params={"waitUntilComplete": True}, data = command_payload)
        
        if verbose == True:
            self.display_responce(r)

    def get_all_runs(self) -> requests.models.Response:
        """Get a responce from robot's server with the information of the last 20 runs.

        Returns:
            requests.models.Response: responce object from the robot's server.
        """        
        r = self.get(self.runs_url, self.HEADERS)
        return r
    
    def get_run_info(self) -> dict:
        """Gets information on the current run (if any), and simple statistics on how many runs exited with which status.

        Returns:
            dict: dictionary detailed info about the runs. 
        """        
        all_runs_json = self.get_all_runs()
        all_runs_json = json.loads(all_runs_json.text)
        data = all_runs_json['data']
        current_run_id = None
        current_run_status = None
        for run in data:
            if run['current'] == True:
                current_run_id = run['id']
                current_run_status = run['status']
                self.run_id = current_run_id

        print(f"Total number of runs: {all_runs_json['meta']['totalLength']}")
        print(f"Current run ID: {current_run_id}")
        print(f"Current run status: {current_run_status}")
        return data