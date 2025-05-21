from opentrons import protocol_api, types
import os, sys
upstream_dir = os.path.abspath(os.path.join(os.getcwd(), '.'))
if upstream_dir not in sys.path:
    sys.path.insert(0, upstream_dir)
print(upstream_dir)
from microtissue_manipulator import core, utils
import cv2


# metadata
metadata = {
    "protocolName": "Testing custom pipettes",
    "author": "Ivan",
    "description": "Testing cuboid pipettes for liquid handling",
}

# requirements
requirements = {"robotType": "OT-2", "apiLevel": "2.19"}

# protocol run function
def run(protocol: protocol_api.ProtocolContext):
    # labware
    cap = core.Camera(0)

    tiprack = protocol.load_labware(
        "vwr_96_tiprack_200ul", location="7"
    )
    well_plate = protocol.load_labware(
        "corning_96_wellplate_360ul_flat", location = "6"
    )
    # pipettes
    left_pipette = protocol.load_instrument(
        "p300_single", mount="left", tip_racks=[tiprack]
    )
    # commands

    left_pipette.flow_rate.aspirate = 50
    left_pipette.flow_rate.dispense = 100
    left_pipette.flow_rate.blow_out = 75

    left_pipette.pick_up_tip()


    window = cap.get_window()
    while True:
        frame = cap.get_frame(undist=True)

        cv2.imshow(cap.window_name, frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()

    loc = types.Location(types.Point(x = 200, y = 50, z = 5), labware=None)

    left_pipette.move_to(loc)
    left_pipette.aspirate(50)

    left_pipette.dispense(50, location = well_plate["A1"])
    left_pipette.drop_tip(location=tiprack["A1"])

    
