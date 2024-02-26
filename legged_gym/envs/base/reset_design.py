import tempfile

import numpy as np
import xmltodict
from legged_gym import LEGGED_GYM_ROOT_DIR


class ModifyURDF():
    def __init__(self,original_filepath=None):
        # self.original_urdf_filepath = "assets" + "/" + original_filepath
        self.original_urdf_filepath = LEGGED_GYM_ROOT_DIR + "/" + "legged_gym" + "/" + original_filepath
        self.ori_upper_length = 0.16
        self.ori_lower_length = 0.18
        self.ori_knee_pos = -0.18
        self.ori_ankle_pos = -0.21

        self.ori_thigh_mass = 0.61
        self.ori_shank_mass = 0.115
        self.thigh_xy_length = 0.02
        self.shank_xy_length = 0.02
        self.thigh_z_length = 0.16
        self.shank_z_length = 0.18

        self.ori_thigh_origin_z = -0.0273
        self.ori_shank_origin_z = -0.12

        self.FL_THIGH_IDX = 4
        self.FL_SHANK_IDX = 5
        self.FR_THIGH_IDX = 9
        self.FR_SHANK_IDX = 10
        self.HL_THIGH_IDX = 14
        self.HL_SHANK_IDX = 15
        self.HR_THIGH_IDX = 19
        self.HR_SHANK_IDX = 20

        self.FL_Knee_joint_idx = 4
        self.FL_Ankle_idx = 5
        self.FR_Knee_joint_idx = 9
        self.FR_Ankle_idx = 10
        self.HL_Knee_joint_idx = 14
        self.HL_Ankle_idx = 15
        self.HR_Knee_joint_idx = 19
        self.HR_Ankle_idx = 20

        self.scale = 0.4
        self.range_high = 1.0 + self.scale
        self.range_low = 1.0 - self.scale

    def calc_inertia(self,type=None,factor=None):
        if type == "thigh":
            ixx = (self.ori_thigh_mass * factor / 12) * (self.thigh_xy_length ** 2 + (self.thigh_z_length * factor) ** 2)
            iyy = (self.ori_thigh_mass * factor / 12) * (self.thigh_xy_length ** 2 + (self.thigh_z_length * factor) ** 2)
            izz = (self.ori_thigh_mass * factor / 12) * (self.thigh_xy_length ** 2 + self.thigh_xy_length ** 2)
            return ixx,iyy,izz
        elif type == "shank":
            ixx = (self.ori_shank_mass * factor / 12) * (self.shank_xy_length ** 2 + (self.shank_z_length * factor) ** 2)
            iyy = (self.ori_shank_mass * factor / 12) * (self.shank_xy_length ** 2 + (self.shank_z_length * factor) ** 2)
            izz = (self.ori_shank_mass * factor / 12) * (self.shank_xy_length ** 2 + self.shank_xy_length ** 2)
            return ixx,iyy,izz
        else:
            print("Please specify the leg type !!!")
            raise NotImplementedError


    def reset_design(self,design=None):
        front_upper_ratio = design[0]
        front_lower_ratio = design[1]
        hind_upper_ratio = design[2]
        hind_lower_ratio = design[3]

        '''create tempory file'''
        new_urdf_file = tempfile.NamedTemporaryFile(delete=False,prefix="Quadruped_",suffix=".urdf")
        new_urdf_filepath = new_urdf_file.name
        '''modify the design'''
        with open(self.original_urdf_filepath,'r') as fd:
            xml_string = fd.read()
            xml_dict = xmltodict.parse(xml_string)

            ##### FL_leg and FR_leg#####
            # FL_THIGH and FR_THIGH
            for idx in {self.FL_THIGH_IDX,self.FR_THIGH_IDX}:
                # xml_dict['robot']['link'][idx]['inertial']['origin']['@xyz'] = "-0.00523 -0.0216 {}".format(self.ori_thigh_origin_z * front_upper_ratio)
                xml_dict['robot']['link'][idx]['inertial']['origin']['@xyz'] = "0 0 {}".format( (- self.ori_upper_length / 2) * front_upper_ratio)
                xml_dict['robot']['link'][idx]['inertial']['mass']['@value'] = self.ori_thigh_mass * front_upper_ratio
                xml_dict['robot']['link'][idx]['inertial']['inertia']['@ixx'] = self.calc_inertia(type='thigh',factor=front_upper_ratio)[0]
                xml_dict['robot']['link'][idx]['inertial']['inertia']['@iyy'] = self.calc_inertia(type='thigh', factor=front_upper_ratio)[1]
                xml_dict['robot']['link'][idx]['inertial']['inertia']['@izz'] = self.calc_inertia(type='thigh', factor=front_upper_ratio)[2]
                xml_dict['robot']['link'][idx]['visual']['origin']['@xyz'] = "0 0 {}".format( (- self.ori_upper_length / 2) * front_upper_ratio)
                xml_dict['robot']['link'][idx]['visual']['geometry']['box']['@size'] = "0.02 0.02 {}".format(self.ori_upper_length * front_upper_ratio)
                xml_dict['robot']['link'][idx]['collision']['origin']['@xyz'] = "0 0 {}".format( (- self.ori_upper_length / 2) * front_upper_ratio)
                xml_dict['robot']['link'][idx]['collision']['geometry']['box']['@size'] = "0.02 0.02 {}".format(self.ori_upper_length * front_upper_ratio)
            # FL_Knee_joint and FR_Knee_joint
            for idx in {self.FL_Knee_joint_idx,self.FR_Knee_joint_idx}:
                xml_dict['robot']['joint'][idx]['origin']['@xyz'] = "0 0 {}".format(self.ori_knee_pos * front_upper_ratio)
            # FL_SHANK and FR_SHANK
            for idx in {self.FL_SHANK_IDX,self.FR_SHANK_IDX}:
                # xml_dict['robot']['link'][idx]['inertial']['origin']['@xyz'] = "0.00585 -8.732E-07 {}".format(self.ori_shank_origin_z * front_lower_ratio)
                xml_dict['robot']['link'][idx]['inertial']['origin']['@xyz'] = "0 0 {}".format((-self.ori_lower_length / 2) * front_lower_ratio)
                xml_dict['robot']['link'][idx]['inertial']['mass']['@value'] = self.ori_shank_mass * front_lower_ratio
                xml_dict['robot']['link'][idx]['inertial']['inertia']['@ixx'] = self.calc_inertia(type="shank",factor=front_lower_ratio)[0]
                xml_dict['robot']['link'][idx]['inertial']['inertia']['@iyy'] = self.calc_inertia(type="shank", factor=front_lower_ratio)[1]
                xml_dict['robot']['link'][idx]['inertial']['inertia']['@izz'] = self.calc_inertia(type="shank", factor=front_lower_ratio)[2]
                xml_dict['robot']['link'][idx]['visual']['origin']['@xyz'] = "0 0 {}".format((-self.ori_lower_length / 2) * front_lower_ratio)
                xml_dict['robot']['link'][idx]['visual']['geometry']['box']['@size'] = "0.02 0.02 {}".format(self.ori_lower_length * front_lower_ratio)
                xml_dict['robot']['link'][idx]['collision']['origin']['@xyz'] = "0 0 {}".format((-self.ori_lower_length / 2) * front_lower_ratio)
                xml_dict['robot']['link'][idx]['collision']['geometry']['box']['@size'] = "0.02 0.02 {}".format(self.ori_lower_length * front_lower_ratio)
            # FL_Ankle
            for idx in {self.FL_Ankle_idx,self.FR_Ankle_idx}:
                xml_dict['robot']['joint'][idx]['origin']['@xyz'] = "0 0 {}".format(self.ori_ankle_pos * front_lower_ratio)


            ##### HL_leg and HR_leg #####
            # HL_THIGH
            for idx in {self.HL_THIGH_IDX,self.HR_THIGH_IDX}:
                # xml_dict['robot']['link'][idx]['inertial']['origin']['@xyz'] = "-0.00523 -0.0216 {}".format(self.ori_thigh_origin_z * hind_upper_ratio)
                xml_dict['robot']['link'][idx]['inertial']['origin']['@xyz'] = "0 0 {}".format( (- self.ori_upper_length / 2) * hind_upper_ratio)
                xml_dict['robot']['link'][idx]['inertial']['mass']['@value'] = self.ori_thigh_mass * hind_upper_ratio
                xml_dict['robot']['link'][idx]['inertial']['inertia']['@ixx'] = self.calc_inertia(type='thigh', factor=hind_upper_ratio)[0]
                xml_dict['robot']['link'][idx]['inertial']['inertia']['@iyy'] = self.calc_inertia(type='thigh', factor=hind_upper_ratio)[1]
                xml_dict['robot']['link'][idx]['inertial']['inertia']['@izz'] = self.calc_inertia(type='thigh', factor=hind_upper_ratio)[2]
                xml_dict['robot']['link'][idx]['visual']['origin']['@xyz'] = "0 0 {}".format( (- self.ori_upper_length / 2) * hind_upper_ratio)
                xml_dict['robot']['link'][idx]['visual']['geometry']['box']['@size'] = "0.02 0.02 {}".format(self.ori_upper_length * hind_upper_ratio)
                xml_dict['robot']['link'][idx]['collision']['origin']['@xyz'] = "0 0 {}".format( (- self.ori_upper_length / 2) * hind_upper_ratio)
                xml_dict['robot']['link'][idx]['collision']['geometry']['box']['@size'] = "0.02 0.02 {}".format(self.ori_upper_length * hind_upper_ratio)
            # HL_Knee_joint
            for idx in {self.HL_Knee_joint_idx,self.HR_Knee_joint_idx}:
                xml_dict['robot']['joint'][idx]['origin']['@xyz'] = "0 0 {}".format(self.ori_knee_pos * hind_upper_ratio)
            # HL_SHANK
            for idx in {self.HL_SHANK_IDX,self.HR_SHANK_IDX}:
                # xml_dict['robot']['link'][idx]['inertial']['origin']['@xyz'] = "0.00585 -8.732E-07 {}".format(self.ori_shank_origin_z * hind_lower_ratio)
                xml_dict['robot']['link'][idx]['inertial']['origin']['@xyz'] = "0 0 {}".format((-self.ori_lower_length / 2) * hind_lower_ratio)
                xml_dict['robot']['link'][idx]['inertial']['mass']['@value'] = self.ori_shank_mass * hind_lower_ratio
                xml_dict['robot']['link'][idx]['inertial']['inertia']['@ixx'] = self.calc_inertia(type="shank", factor=hind_lower_ratio)[0]
                xml_dict['robot']['link'][idx]['inertial']['inertia']['@iyy'] = self.calc_inertia(type="shank", factor=hind_lower_ratio)[1]
                xml_dict['robot']['link'][idx]['inertial']['inertia']['@izz'] = self.calc_inertia(type="shank", factor=hind_lower_ratio)[2]
                xml_dict['robot']['link'][idx]['visual']['origin']['@xyz'] = "0 0 {}".format((-self.ori_lower_length / 2) * hind_lower_ratio)
                xml_dict['robot']['link'][idx]['visual']['geometry']['box']['@size'] = "0.02 0.02 {}".format(self.ori_lower_length * hind_lower_ratio)
                xml_dict['robot']['link'][idx]['collision']['origin']['@xyz'] = "0 0 {}".format((-self.ori_lower_length / 2) * hind_lower_ratio)
                xml_dict['robot']['link'][idx]['collision']['geometry']['box']['@size'] = "0.02 0.02 {}".format(self.ori_lower_length * hind_lower_ratio)
            # HL_Ankle
            for idx in {self.HL_Ankle_idx,self.HR_Ankle_idx}:
                xml_dict['robot']['joint'][idx]['origin']['@xyz'] = "0 0 {}".format(self.ori_ankle_pos * hind_lower_ratio)

            xml_string = xmltodict.unparse(xml_dict,pretty=True)
            with open(new_urdf_filepath,'w') as fd:
                fd.write(xml_string)

        return new_urdf_filepath

    def randomize_design(self):
        front_upper_scale = np.random.uniform(self.range_low,self.range_high,1)[0]
        front_lower_scale = np.random.uniform(self.range_low,self.range_high,1)[0]
        hind_upper_scale = np.random.uniform(self.range_low,self.range_high,1)[0]
        hind_lower_scale = np.random.uniform(self.range_low,self.range_high,1)[0]
        random_design = np.array([front_upper_scale,front_lower_scale,hind_upper_scale,hind_lower_scale])
        new_urdf_filepath = self.reset_design(random_design)

        return new_urdf_filepath,random_design

    def specific_design(self,scales):
        front_upper_scale = scales[0]
        front_lower_scale = scales[1]
        hind_upper_scale = scales[2]
        hind_lower_scale = scales[3]
        specific_design = np.array([front_upper_scale,front_lower_scale,hind_upper_scale,hind_lower_scale])
        new_urdf_filepath = self.reset_design(specific_design)

        return new_urdf_filepath,specific_design








