import numpy as np
import lxml
from  lxml import etree
import re


def array_to_string(array):
    return " ".join([str(x) for x in array])

# common
finger_keys = ["ff", "mf", "rf", "th"]
knuckle_keys = ["fingertip"]
panel_center_and_offset = {
    # "xfpip": ([0, 0.0065, 0.016], -0.0135),
    # "xfdip": ([0, 0.007, 0.0135], -0.016),
    "xffingertip": ([-0.0123, -0.035, 0.0147], 0.0),
    # "thpip": ([0.009, 0, 0.018], -0.0185),
    # "thdip": ([0.009, 0, 0.018], -0.018),
    "thfingertip": ([-0.0123, -0.045, -0.0147], 0.0)
}
taxel_spacing = 0.003
taxel_size = (4, 4)
taxel_radius = 0.001
taxel_rgba = [0.2, 0.2, 0.7, 0.5]
taxel_mass = 0.00001
taxel_diaginertia = [0.000001, 0.000001, 0.000001]
taxel_stiffness = 1000.

# parse original chain
# tree and root for Mujoco sim.
tree = etree.parse("right_hand.xml")
root = tree.getroot()

# kin_tree and kin_root for pytorch_kinematics
kin_root_new = etree.Element("worldbody")
kin_root_new.extend(etree.parse("right_hand.xml").getroot())
kin_root = etree.Element("mujoco")
kin_root.append(kin_root_new)
# delete all geom tags
for geom in kin_root.findall('.//geom'):
    parent = geom.getparent()  # 获得父标签
    parent.remove(geom)        # 从父标签中删除 geom
# delete all childclass attributes
for elem in kin_root.iter():
    if 'childclass' in elem.attrib:
        del elem.attrib['childclass']
# set joint axis
pattern = re.compile(r'^\d+$')
for joint in kin_root.iter('joint'):
    if 'name' in joint.attrib and pattern.match(joint.attrib['name']):
        joint.set('axis', '0 0 -1')

# create sensor for taxel readings sim.
sensor_root = etree.Element("mujocoinclude")
sensor_family = etree.SubElement(sensor_root, "sensor")

def add_taxel_to_hand(hand: etree.Element, hand_kin: etree.Element):
    for finger in finger_keys:
        add_taxel_to_finger(hand, hand_kin, finger)

def add_taxel_to_finger(hand: etree.Element, hand_kin: etree.Element, finger: str):
    for name in knuckle_keys:
        knuckle_key = f'{finger}{name}'
        knuckle = hand.findall(f".//body[@name='{knuckle_key}']")[0]
        knuckle_kin = hand_kin.findall(f".//body[@name='{knuckle_key}']")[0]
        if finger == 'th':
            taxel_offset_axis = 0
        else:
            taxel_offset_axis = 1
        add_taxel_to_knuckle(knuckle, knuckle_kin, f'{finger}{name}', taxel_offset_axis)

def add_taxel_to_knuckle(knuckle: etree.Element, knuckle_kin: etree.Element, knuckle_key:str, offset_axis=1):
    taxel_hori_num = taxel_size[0]
    taxel_vert_num = taxel_size[1]
    # collision filtering
    # knuckle_geom = knuckle.findall(f".//geom[@name='C_{knuckle_key}']")[0]
    # knuckle_geom.set('contype', '1')
    # knuckle_geom.set('conaffinity', '2')

    if knuckle_key[:2] == "th":
        base_pos, offset = panel_center_and_offset[knuckle_key]
    else:
        base_pos, offset = panel_center_and_offset[f"xf{knuckle_key[2:]}"]
    offset -= 0.0
    if offset_axis == 0:
        mount_pos = np.array(base_pos) + np.array([offset, 0, 0])
        mount_quat = np.array([0.70710678, 0., 0.70710678, 0.])
    elif offset_axis == 1:
        mount_pos = np.array(base_pos) + np.array([0, offset, 0])
        mount_quat = np.array([0.70710678, 0., 0.70710678, 0.])
    panel_base = etree.SubElement(knuckle, "body", \
                                  attrib={'name':f'{knuckle_key}_panel_base', 'pos':array_to_string(mount_pos), 'quat':array_to_string(mount_quat)})
    panel_base_kin = etree.SubElement(knuckle_kin, "body", \
                                      attrib={'name':f'{knuckle_key}_panel_base', 'pos':array_to_string(mount_pos), 'quat':array_to_string(mount_quat)})

    for i in range(taxel_hori_num):
        for j in range(taxel_vert_num):
            # if offset_axis == 0:
            #     pos = mount_pos + \
            #         np.array([0.0, 0.5*taxel_spacing, 0.5*taxel_spacing]) + \
            #         np.array([0.0, -(taxel_vert_num//2)*taxel_spacing, -(taxel_hori_num//2)*taxel_spacing]) + \
            #         np.array([0.0, j*taxel_spacing, i*taxel_spacing])
            #     taxel_joint_axis = np.array([1, 0, 0])
            # elif offset_axis == 1:
            #     pos = mount_pos + \
            #         np.array([0.5*taxel_spacing, 0.0, 0.5*taxel_spacing]) + \
            #         np.array([-(taxel_vert_num//2)*taxel_spacing, 0.0, -(taxel_hori_num//2)*taxel_spacing]) + \
            #         np.array([j*taxel_spacing, 0.0, i*taxel_spacing])
            #     taxel_joint_axis = np.array([0, 1, 0])

            pos = np.array([0.5*taxel_spacing, 0.5*taxel_spacing, 0.0]) + \
                np.array([-(taxel_vert_num//2)*taxel_spacing, -(taxel_hori_num//2)*taxel_spacing, 0.0]) + \
                np.array([j*taxel_spacing, i*taxel_spacing, 0.0])
            taxel_joint_axis = np.array([0, 0, 1])
            
            # taxel = etree.SubElement(knuckle, "body", \
            #                             attrib={'name':f'{knuckle_key}_T_r{i}c{j}', 'pos':array_to_string(pos), 'quat':array_to_string(quat)})
            taxel = etree.SubElement(panel_base, "body", \
                                        attrib={'name':f'{knuckle_key}_T_r{i}c{j}', 'pos':array_to_string(pos)})
            taxel_geom = etree.SubElement(taxel, "geom", \
                                        attrib={'type':'sphere', 'size':str(taxel_radius), 'rgba':array_to_string(taxel_rgba), \
                                                # 'contype':'1', 'conaffinity':'2', 'class':'DC_Taxel'})
                                                'class':'DC_Taxel'})
            taxel_inertial = etree.SubElement(taxel, "inertial", \
                                            attrib={'mass':str(taxel_mass), 'diaginertia':array_to_string(taxel_diaginertia), 'pos':'0 0 0'})
            taxel_joint = etree.SubElement(taxel, "joint", \
                                        attrib={'type':'slide', 'stiffness':str(taxel_stiffness), 'damping':'0.0001', \
                                                'axis':array_to_string(taxel_joint_axis), 'name':f'{knuckle_key}_J_r{i}c{j}', 'range':'-0.001 0.001'})
            taxel_sensor_site = etree.SubElement(taxel, "site", \
                                                 attrib={'name':f'{knuckle_key}_S_r{i}c{j}', 'pos':'0 0 0', 'class':'D_Array'})
            # declare sensor
            taxel_sensor = etree.SubElement(sensor_family, "touch", \
                                            attrib={'name':f'{knuckle_key}_S_r{i}c{j}', 'site':f'{knuckle_key}_S_r{i}c{j}', 'cutoff':'1.0'})

def add_or_remove_taxel_from_panel(mjcf_path):
    panel_taxel_size = taxel_size
    panel_taxel_spacing = taxel_spacing
    panel_taxel_offset = -0.003

    panel_xml = etree.parse(mjcf_path)
    panel_root = panel_xml.getroot()

    panel_taxel_hori_num = panel_taxel_size[0]
    panel_taxel_vert_num = panel_taxel_size[1]

    panel_body = panel_root.findall(f".//body[@name='panel_base']")[0]
    panel_body_geom = panel_body.findall(f".//geom[@name='panel_base_geom']")[0]
    panel_body_geom.set('contype', '1')
    panel_body_geom.set('conaffinity', '2')
    panel_sensor = etree.SubElement(panel_root, "sensor")

    mount_pos = np.zeros(3,) + np.array([0.0, 0.0, panel_taxel_offset])
    panel_base = etree.SubElement(panel_body, "site", attrib={'name':'panel_base', 'pos':array_to_string(mount_pos)})

    for i in range(panel_taxel_hori_num):
        for j in range(panel_taxel_vert_num):
            pos = mount_pos + \
                np.array([0.5*panel_taxel_spacing, 0.5*panel_taxel_spacing, 0.0]) + \
                np.array([-(panel_taxel_vert_num//2)*panel_taxel_spacing, -(panel_taxel_hori_num//2)*panel_taxel_spacing, 0.0]) + \
                np.array([j*panel_taxel_spacing, i*panel_taxel_spacing, 0.0])
            taxel = etree.SubElement(panel_body, "body", \
                                        attrib={'name':f'panel_T_r{i}c{j}', 'pos':array_to_string(pos)})
            taxel_geom = etree.SubElement(taxel, "geom", \
                                        attrib={'type':'sphere', 'size':str(taxel_radius), 'rgba':array_to_string(taxel_rgba), \
                                                'contype':'1', 'conaffinity':'2'})
            taxel_inertial = etree.SubElement(taxel, "inertial", \
                                            attrib={'mass':str(taxel_mass), 'diaginertia':array_to_string(taxel_diaginertia), 'pos':'0 0 0'})
            taxel_joint = etree.SubElement(taxel, "joint", \
                                        attrib={'type':'slide', 'stiffness':str(taxel_stiffness), 'damping':'0.0001', \
                                                'axis':'0 0 1', 'name':f'panel_J_r{i}c{j}', 'range':'-0.001 0.001'})
            taxel_sensor_site = etree.SubElement(taxel, "site", \
                                                 attrib={'name':f'panel_S_r{i}c{j}', 'pos':'0 0 0', 'class':'D_Array'})
            # declare sensor
            taxel_sensor = etree.SubElement(panel_sensor, "touch", \
                                            attrib={'name':f'panel_S_r{i}c{j}', 'site':f'panel_S_r{i}c{j}', 'cutoff':'1.0'})

    panel_tree = etree.ElementTree(panel_root)
    panel_tree.write("./Adroit/my_sensor_panel_v2.xml", pretty_print=True, xml_declaration=False, encoding='utf-8')

"""
    The following lines generate three files
    1. Adroit/resources/chain_with_touch.xml, which is the hand model with taxel sensors,
       this file keeps the same structure as Adroit/resources/chain_with_touch.xml
    2. Adroit/Adroit_hand_kin_v2.xml, which is the hand model with panel base only without
       taxels, this file keeps the same structure as Adroit/Adroit_hand_kin.xml
    3. Adroit/resources/touch_sensor_array.xml, which is the sensor declaration for
       all taxels in Adroit/resources/chain_with_touch.xml

    All three files are consistent with taxel names and kinematics

    Note that 1.chain_with_touch.xml and 3.touch_sensor_array.xml are together included
    in Adroit/Adroit_hand.xml. While Adroit_hand_kin_v2.xml is used for kinematics only
"""
# add taxels
add_taxel_to_hand(root, kin_root)

# save new chain
tree = etree.ElementTree(root)
kin_tree = etree.ElementTree(kin_root)
sensor_tree = etree.ElementTree(sensor_root)
tree.write('right_hand_with_touch.xml', pretty_print=True, xml_declaration=False, encoding='utf-8')
kin_tree.write('Leap_hand_kin.xml', pretty_print=True, xml_declaration=False, encoding='utf-8')
sensor_tree.write('touch_sensor_array.xml', pretty_print=True, xml_declaration=False, encoding='utf-8')

"""
    The following lines generate file my_sensor_panel.xml, which is a standalone taxel
    panel used for data collection.
    This file is consistent with all three files above in
        - taxel size (i.e., 4x4)
        - taxel inertia
        - taxel sphere collision body radius
        - taxel joint stiffness and damping
"""
# add_or_remove_taxel_from_panel(mjcf_path="./Adroit/my_sensor_panel.xml")
