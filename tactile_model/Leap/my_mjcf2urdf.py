import xml
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom


def has_joint_attib(body: ET.Element):
    for child in list(body):
        if child.tag == "joint":
            return True
    return False

def get_joint(body: ET.Element):
    for child in list(body):
        if child.tag == "joint":
            return child
    return None

def mjcf_to_urdf(mjcf_str):
    # 解析MJCF字符串
    mjcf_root = ET.fromstring(mjcf_str)

    # 创建URDF的根元素
    urdf_root = ET.Element("robot")
    urdf_root.set("name", "converted_robot")

    all_joints_meta = {}

    base_joint = ET.SubElement(urdf_root, "link")
    base_joint.set("name", "world")

    # 遍历MJCF文件中的body元素
    for body in mjcf_root.findall(".//body"):
        if body.get("name") in ["forearm", "palm_lower"]:
            base_joint = ET.SubElement(urdf_root, "joint")
            base_joint.set("name", "base_joint")
            base_joint.set("type", "fixed")
            _parent = ET.SubElement(base_joint, "parent")
            _parent.set("link", "world")
            _child = ET.SubElement(base_joint, "child")
            _child.set("link", body.get("name"))
            origin = ET.SubElement(base_joint, "origin")
            origin.set("xyz", body.get("pos"))
            origin.set("rpy", quat_to_rpy(body.get("quat")))

        if body.get("name").endswith("panel_base"):
            ghost_link = ET.SubElement(urdf_root, "link")
            ghost_link.set("name", body.get("name"))

            ghost_joint = ET.SubElement(urdf_root, "joint")
            origin = ET.SubElement(ghost_joint, "origin")
            origin.set("xyz", body.get("pos"))
            origin.set("rpy", quat_to_rpy(body.get("quat")))
            ghost_joint.set("name", body.get("name") + "_joint")
            ghost_joint.set("type", "fixed")
            _parent = ET.SubElement(ghost_joint, "parent")
            _parent.set("link", body.get("name").split("_")[0])
            _child = ET.SubElement(ghost_joint, "child")
            _child.set("link", body.get("name"))
            continue

        for child in list(body):
            if has_joint_attib(child):
                all_joints_meta[get_joint(child).get("name")] = {
                    "parent": body.get("name"),
                    "child": child.get("name"),
                    "xyz": child.get("pos") if child.get("pos") else "0 0 0",
                    "rpy": quat_to_rpy(child.get("quat")) if child.get("quat") else "0 0 0"
                }

        # 创建URDF的link元素
        link = ET.SubElement(urdf_root, "link")
        link.set("name", body.get("name"))

        # 转换body的位置和四元数
        # if body.get("pos"):
        #     origin = ET.SubElement(link, "origin")
        #     origin.set("xyz", body.get("pos"))
        # if body.get("quat"):
        #     origin.set("rpy", quat_to_rpy(body.get("quat")))

    print(f"Found {len(all_joints_meta)} joints with meta: ", all_joints_meta)

    # 遍历MJCF文件中的joint元素
    for joint in mjcf_root.findall(".//joint"):
        # 创建URDF的joint元素
        urdf_joint = ET.SubElement(urdf_root, "joint")
        urdf_joint.set("name", joint.get("name"))

        # 转换body的位置和四元数
        xyz = all_joints_meta[joint.get("name")]["xyz"]
        rpy = all_joints_meta[joint.get("name")]["rpy"]
        origin = ET.SubElement(urdf_joint, "origin")
        origin.set("xyz", xyz)
        origin.set("rpy", rpy)

        parent_link = all_joints_meta[joint.get("name")]["parent"]
        child_link = all_joints_meta[joint.get("name")]["child"]
        parent = ET.SubElement(urdf_joint, "parent")
        parent.set("link", parent_link)
        child = ET.SubElement(urdf_joint, "child")
        child.set("link", child_link)

        # 转换关节类型
        joint_type = joint.get("type")
        if joint_type == "slide":
            urdf_joint.set("type", "prismatic")
        else:
            urdf_joint.set("type", "revolute")

        # 转换位置和轴
        # if joint.get("pos"):
        #     origin = ET.SubElement(urdf_joint, "origin")
        #     origin.set("xyz", joint.get("pos"))
        axis = ET.SubElement(urdf_joint, "axis")
        if joint.get("axis"):
            axis.set("xyz", joint.get("axis"))
        else:
            axis.set("xyz", "0 0 1")

        # joint limits
        limits = ET.SubElement(urdf_joint, "limit")
        limits.set("effort", "30")
        limits.set("velocity", "1.0")
        limits.set("lower", joint.get("range").split()[0])
        limits.set("upper", joint.get("range").split()[1])


    # 将URDF的树结构转换为字符串
    urdf_str = ET.tostring(urdf_root, encoding="unicode")
    return urdf_str

def quat_to_rpy(quat_str):
    def quat_wxyz_to_xyzw(quat):
        return [quat[1], quat[2], quat[3], quat[0]]
    
    # 将四元数转换为RPY欧拉角
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    quat = list(map(float, quat_str.split()))
    r = R.from_quat(quat_wxyz_to_xyzw(quat)) # wxyz -> xyzw
    rpy = r.as_euler('xyz', degrees=False)
    return " ".join(map(str, rpy))

# 示例MJCF字符串
mjcf_str = open("./Leap_hand_kin.xml").read()

def format_xml_string(xml_str):
    # 解析XML字符串
    parsed_str = minidom.parseString(xml_str)
    # 格式化并缩进
    pretty_str = parsed_str.toprettyxml(indent="  ")
    # 去除多余的空行
    return "\n".join([line for line in pretty_str.splitlines() if line.strip()])

def save_urdf_to_file(urdf_str, file_name="converted_robot.urdf"):
    # 格式化URDF字符串
    formatted_str = format_xml_string(urdf_str)
    # 打开文件并写入格式化后的URDF字符串
    with open(file_name, "w") as file:
        file.write(formatted_str)

# 转换为URDF
urdf_str = mjcf_to_urdf(mjcf_str)
save_urdf_to_file(urdf_str, "Leap_hand_kin.urdf")
