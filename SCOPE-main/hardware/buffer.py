import os
import math
import configparser
import pdb
from general.logger import log_print
# from general.parameters import *
from general.param_runtime import P


class BufferBase:
    def __init__(self, depth, width):
        self.depth = depth
        self.width = width * P.ACTIVATION_BIT
        self.device_tech = P.DEVICE_TECH
        self.queue = []
        self.reads = 0
        self.writes = 0
        self.total_times = 0
        self.area = 0
        self.power = 0
        self.energy = 0
        self.show_buffer = True
        
        # Each subclass should implement its own capacity calculation
        self.capacity = self._calculate_capacity()

    def _calculate_capacity(self):
        raise NotImplementedError("Implement this method in subclasses")

    def is_empty(self):
        return len(self.queue) == 0

    def is_full(self):
        return len(self.queue) == self.capacity

    def peek(self):
        if self.is_empty():
            raise IndexError("Buffer is empty")
        return self.queue[0]

    def size(self):
        return len(self.queue)

    def __str__(self):
        if self.show_buffer:
            return f"{self.__class__.__name__}({self.queue})"
        return self.__class__.__name__

class Global_buffer(object):
    def __init__(self, Hw_config,channel_size, buffer_level):

        self.Hw_config = Hw_config
        self.channel_size = channel_size
        self.buffer_level = buffer_level
        self.area = 0 

        self.show_buffer = True
        self.writes = 0
        self.reads = 0


        if buffer_level == 1:        
            self.area_config = float(Hw_config.get('Architecture level', 'Buffer_area', fallback =0))
        elif buffer_level == 2:
            self.area_config = float(Hw_config.get('Sensor level', 'Buffer_area', fallback =0))


        self.device_tech = P.DEVICE_TECH

        self.buf_type = int(Hw_config.get('Architecture level', 'Buffer_Type', fallback = 1))
        self.capacity = int(Hw_config.get('Architecture level', 'Buffer_Capacity'))
        self.wordwidth = int(Hw_config.get('Architecture level', 'Buffer_Wordwidth'))
        self.buffer_num = int(Hw_config.get('Architecture level', 'Buffer_num', fallback = 1))

        

        self.size = self.capacity *self.buffer_num * 1024*8 /self.wordwidth #4096
        self.buffer = [] 

        self.get_performance_basic()



    def get_performance_basic(self):      

        
        if self.buf_type == 1:
            area_dict = {
                # ProcessingNode, Capacity, WordWidth
                (180, 2, 32): (4.542, 2.267, 15.548, 944.568, 944.568, 114912.521),   # (ReadPower, WritePower, LeakagePower,
                (180, 2, 64): (4.476, 3.029, 15.477, 933.351, 933.351, 113970.811),  #  ReadLatency, WriteLatency, Area)
                (180, 2, 128): (6.586, 3.738, 14.951, 950.514, 950.514, 112759.281),  # pj, pj, uW, ps, ps, um^2
                (180, 2, 256): (12.532, 6.835, 14.951, 950.514, 950.514, 112759.281),

                (180, 4, 32): (8.483, 3.153, 28.468, 1.051, 1.051, 211253.834),  # pj, pj, uW, ns, ns, um^2
                (180, 4, 64): (8.482, 3.980, 28.247, 1.040, 1.040, 211136.467),
                (180, 4, 128): (8.359, 5.510, 28.136, 1.029, 1.029, 209274.469),
                (180, 4, 256): (15.952, 10.256, 28.136, 1.029, 1.029, 209274.469),

                (180, 8, 32): (12.042, 4.240, 54.922, 1.331, 1.331, 402979.451),  # pj, pj, uW, ns, ns, um^2
                (180, 8, 64): (16.207, 5.678, 52.698, 1.195, 1.195, 396765.816),
                (180, 8, 128): (16.206, 7.332, 52.351, 1.184, 1.184, 396656.494),
                (180, 8, 256): (15.965, 10.400, 52.176, 1.173, 1.173, 393181.229),

                (180, 16, 32): (23.055, 5.932, 102.424, 1.493, 1.493, 764706.067),  # pj, pj, uW, ns, ns, um^2
                (180, 16, 64): (23.054, 7.582, 101.731, 1.472, 1.472, 764385.904),
                (180, 16, 128): (23.053, 10.883, 101.384, 1.461, 1.461, 764232.409),
                (180, 16, 256): (22.812, 17.248, 101.209, 1.450, 1.450, 760706.454),

                (180, 32, 32): (36.707, 8.049, 200.238, 1.726, 1.726, 1.486),  # pj, pj, uW, ns, ns, mm^2
                (180, 32, 64): (44.533, 10.447, 178.111, 1.739, 1.739, 1.454),
                (180, 32, 128): (44.532, 13.748, 177.221, 1.717, 1.717, 1.454),
                (180, 32, 256): (44.531, 20.352, 176.774, 1.706, 1.706, 1.454),

                (180, 64, 32): (71.293, 10.838, 350.135, 2.029, 2.029, 2.847),  # pj, pj, uW, ns, ns, mm^2
                (180, 64, 64): (71.277, 14.121, 348.324, 1.985, 1.985, 2.847),
                (180, 64, 128): (71.276, 20.718, 347.434, 1.963, 1.963, 2.847),
                (180, 64, 256): (71.275, 33.913, 346.987, 1.952, 1.952, 2.847),

                (180, 128, 32): (124.949, 15.058, 691.429, 2.533, 2.533, 5.624),  # pj, pj, uW, ns, ns, mm^2
                (180, 128, 64): (140.405, 19.835, 657.169, 2.633, 2.633, 5.556),
                (180, 128, 128): (140.393, 26.421, 654.782, 2.590, 2.590, 5.556),
                (180, 128, 256): (140.392, 39.616, 653.603, 2.568, 2.568, 5.556),

                (180, 256, 32): (246.851, 20.814, 1.297, 3.172, 3.172, 11.027),  # pj, pj, mW, ns, ns, mm^2
                (180, 256, 64): (246.622, 27.179, 1.299, 3.166, 3.166, 11.013),
                (180, 256, 128): (246.610, 40.357, 1.296, 3.123, 3.123, 11.013),
                (180, 256, 256): (246.609, 66.735, 1.295, 3.101, 3.101, 11.012),

                (180, 512, 32): (459.247, 28.874, 2.579, 4.058, 4.058, 21.824),  # pj, pj, mW, ns, ns, mm^2
                (180, 512, 64): (459.018, 41.831, 2.580, 4.053, 4.053, 21.810),
                (180, 512, 128): (489.240, 50.696, 2.436, 4.945, 4.945, 21.536),
                (180, 512, 256): (489.228, 77.063, 2.434, 4.902, 4.902, 21.537),


                (65, 2, 32): (0.583, 0.289, 163.835, 501.600, 501.600, 14841.865),  # pj, pj, uW, ps, ps, um^2
                (65, 2, 64): (0.861, 0.387, 159.745, 495.771, 495.771, 14850.340),
                (65, 2, 128): (0.846, 0.481, 158.781, 489.548, 489.548, 14612.433),
                (65, 2, 256): (1.612, 0.881, 158.781, 489.548, 489.548, 14612.433),

                (65, 4, 32): (1.091, 0.402, 304.751, 556.538, 556.538, 27374.069),  # pj, pj, uW, ps, ps, um^2
                (65, 4, 64): (1.090, 0.509, 302.828, 550.485, 550.485, 27358.664),
                (65, 4, 128): (1.076, 0.710, 301.861, 544.284, 544.284, 27117.360),
                (65, 4, 256): (2.057, 1.325, 301.861, 544.284, 544.284, 27117.360),

                (65, 8, 32): (1.553, 0.543, 591.638, 731.545, 731.545, 52281.690),  # pj, pj, uW, ps, ps, um^2
                (65, 8, 64): (2.088, 0.726, 572.531, 630.117, 630.117, 51567.986),
                (65, 8, 128): (2.088, 0.942, 569.498, 624.063, 624.063, 51553.688),
                (65, 8, 256): (2.060, 1.345, 567.979, 617.903, 617.903, 51101.997),

                (65, 16, 32): (2.978, 0.758, 1.119, 816.003, 816.003, 99385.992),  # pj, pj, mW, ps, ps, um^2
                (65, 16, 64): (2.978, 0.973, 1.112, 803.897, 803.897, 99344.115),
                (65, 16, 128): (2.978, 1.403, 1.109, 797.843, 797.843, 99324.126),
                (65, 16, 256): (2.950, 2.235, 1.108, 791.700, 791.700, 98865.878),

                (65, 32, 32): (4.754, 1.032, 2.196, 1.001, 1.001, 192965.950),  # pj, pj, mW, ns, ns, um^2
                (65, 32, 64): (5.759, 1.338, 1.999, 949.602, 949.602, 188945.165),  # pj, pj, mW, ps, ps, um^2
                (65, 32, 128): (5.759, 1.768, 1.991, 937.495, 937.495, 188904.819),
                (65, 32, 256): (5.758, 2.627, 1.987, 931.442, 931.442, 188885.573),

                (65, 64, 32): (9.241, 1.338, 3.943, 1.163, 1.163, 369953.377),  # pj, pj, mW, ns, ns, um^2
                (65, 64, 64): (9.239, 1.815, 3.927, 1.139, 1.139, 369981.762),
                (65, 64, 128): (9.239, 2.674, 3.920, 1.127, 1.127, 369915.295),
                (65, 64, 256): (9.238, 4.392, 3.916, 1.121, 1.121, 369883.589),

                (65, 128, 32): (16.221, 1.934, 7.808, 1.555, 1.555, 731137.882),  # pj, pj, mW, ns, ns, um^2
                (65, 128, 64): (18.214, 2.549, 7.514, 1.508, 1.508, 723906.238),
                (65, 128, 128): (18.212, 3.407, 7.494, 1.483, 1.483, 723930.954),
                (65, 128, 256): (18.212, 5.124, 7.484, 1.471, 1.471, 723873.105),

                (65, 256, 32): (32.063, 2.671, 14.876, 1.916, 1.916, 1.437),  # pj, pj, mW, ns, ns, mm^2
                (65, 256, 64): (32.037, 3.503, 14.889, 1.913, 1.913, 1.435),
                (65, 256, 128): (32.035, 5.219, 14.868, 1.889, 1.889, 1.435),
                (65, 256, 256): (32.035, 8.652, 14.858, 1.877, 1.877, 1.435),

                (65, 512, 32): (59.704, 3.717, 29.615, 2.650, 2.650, 2.843),  # pj, pj, mW, ns, ns, mm^2
                (65, 512, 64): (59.678, 5.407, 29.628, 2.647, 2.647, 2.842),
                (65, 512, 128): (63.578, 6.552, 28.372, 2.962, 2.962, 2.807),
                (65, 512, 256): (63.576, 9.984, 28.371, 2.938, 2.938, 2.807),


                (40, 2, 32): (0.255, 0.128, 80.637, 313.806, 313.806, 5620.588),   # pj, pj, uW, ps, ps, um^2
                (40, 2, 64): (0.375, 0.170, 79.068, 311.315, 311.315, 5631.666),
                (40, 2, 128): (0.369, 0.212, 78.680, 307.492, 307.492, 5541.468),
                (40, 2, 256): (0.702, 0.389, 78.680, 307.492, 307.492, 5541.468),

                (40, 4, 32): (0.478, 0.177, 152.076, 353.739, 353.739, 10366.511),  # pj, pj, uW, ps, ps, um^2
                (40, 4, 64): (0.478, 0.225, 151.301, 350.019, 350.019, 10360.678),
                (40, 4, 128): (0.472, 0.315, 150.911, 346.211, 346.211,  10269.296),
                (40, 4, 256): (0.901, 0.588, 150.911, 346.211, 346.211, 10269.296),  # !! risky,need to verify

                (40, 8, 32): (0.684, 0.239, 296.835, 469.610, 469.610, 19798.983),  # pj, pj, uW, ps, ps, um^2
                (40, 8, 64): (0.915, 0.319, 289.303, 420.340, 420.340, 19593.243),
                (40, 8, 128): (0.915, 0.416, 288.080, 416.620, 416.620, 19587.950),
                (40, 8, 256): (0.903, 0.597, 287.467, 412.835, 412.835, 19416.357),

                (40, 16, 32): (1.313, 0.333, 567.668, 544.460, 544.460, 37732.182),  # pj, pj, uW, ps, ps, um^2
                (40, 16, 64): (1.313, 0.430, 565.231, 537.019, 537.019, 37716.350),
                (40, 16, 128): (1.313, 0.622, 564.008, 533.299, 533.299, 37708.780),
                (40, 16, 256): (1.301, 0.955, 563.394, 529.524, 529.524, 37534.808),

                (40, 32, 32): (2.106, 0.455, 1.18, 679.587, 679.587, 73268.823),  # pj, pj, mW, ps, ps, um^2
                (40, 32, 64): (2.538, 0.587, 1.038, 711.226, 711.226, 711553.199),
                (40, 32, 128): (2.538, 0.780, 1.035, 703.786, 703.786, 711537.919),
                (40, 32, 256): (2.538, 1.164, 1.033, 700.066, 700.066, 711530.618),

                (40, 64, 32): (4.096, 0.609, 2.054, 870.623, 870.623, 140100.687),  # pj, pj, mW, ps, ps, um^2
                (40, 64, 64): (4.095, 0.801, 2.047, 855.742, 855.742, 140111.437),
                (40, 64, 128): (4.095, 1.185, 2.044, 848.302, 848.302, 140086.265),
                (40, 64, 256): (4.095, 1.954, 2.043, 844.581, 844.581, 140074.259),

                (40, 128, 32): (7.219, 0.853, 4.075, 1.161, 1.161, 276880.618),  # pj, pj, mW, ns, ns, um^2
                (40, 128, 64): (7.218, 1.236, 4.069, 1.147, 1.147, 276891.551),
                (40, 128, 128): (8.076, 1.505, 3.952, 1.392, 1.392, 274151.367),
                (40, 128, 256): (8.076, 2.273, 3.948, 1.385, 1.385, 274129.460),

                (40, 256, 32): (14.268, 1.170, 7.864, 1.743, 1.743, 543988.013),  # pj, pj, mW, ns, ns, um^2
                (40, 256, 64): (14.262, 1.548, 7.863, 1.734, 1.734, 543339.029),
                (40, 256, 128): (14.261, 2.315, 7.855, 1.719, 1.719, 543353.978),
                (40, 256, 256): (14.261, 3.852, 7.850, 1.712, 1.712, 543319.008),

                (40, 512, 32): (26.640, 1.642, 15.659, 2.322, 2.322, 1.077),  # pj, pj, mW, ns, ns, mm^2
                (40, 512, 64): (26.629, 2.399, 15.664, 2.320, 2.320, 1.076),
                (40, 512, 128): (26.628, 3.934, 15.656, 2.305, 2.305, 1.076),
                (40, 512, 256): (28.308, 4.434, 15.168, 3.588, 3.588, 1.063),


                (28, 2, 32): (0.136, 0.070, 134.733, 261.797, 261.797, 2786.060),  # pj, pj, uW, ps, ps, um^2
                (28, 2, 64): (0.196, 0.092, 131.715, 259.307, 259.307, 2786.295),
                (28, 2, 128): (0.193, 0.116, 131.022, 256.203, 256.203, 2741.673),
                (28, 2, 256): (0.366, 0.213, 131.022, 256.203, 256.203, 2741.673),

                (28, 4, 32): (0.255, 0.096, 253.420, 299.670, 299.670, 5143.722),  # pj, pj, uW, ps, ps, um^2
                (28, 4, 64): (0.255, 0.123, 252.038, 296.645, 296.645, 5141.005),
                (28, 4, 128): (0.251, 0.175, 251.344, 293.552, 293.552, 5095.698),
                (28, 4, 256): (0.479, 0.326, 251.344, 293.552, 293.552, 5095.698),

                (28, 8, 32): (0.372, 0.131, 494.615, 403.106, 403.106, 9812.488),  # pj, pj, uW, ps, ps, um^2
                (28, 8, 64): (0.712, 0.181, 940.450, 483.415, 483.415, 9607.729),
                (28, 8, 128): (0.486, 0.228, 476.961, 368.984, 368.984, 9605.190),
                (28, 8, 256): (0.480, 0.331, 475.866, 365.910, 365.910, 9521.060),

                (28, 16, 32): (0.346, 0.087, 433.591, 459.989, 459.989, 18518.529),  # pj, pj, uW, ps, ps, um^2
                (28, 16, 64): (0.712, 0.235, 936.098, 477.366, 477.366, 18511.043),
                (28, 16, 128): (0.712, 0.344, 933.915, 474.341, 474.341, 18507.465),
                (28, 16, 256): (0.705, 0.556, 932.817, 471.292, 471.292, 18422.108),

                (28, 32, 32): (1.159, 0.248, 1.851, 618.527, 618.527, 35901.723),  # pj, pj, mW, ps, ps, um^2
                (28, 32, 64): (1.159, 0.357, 1.547, 612.478, 612.478, 35892.528),
                (28, 32, 128): (1.376, 0.427, 1.750, 684.374, 684.374, 35509.337),
                (28, 32, 256): (1.376, 0.645, 1.747, 681.349, 681.349, 35505.760),

                (28, 64, 32): (2.259, 0.331, 3.474, 851.660, 851.660, 69537.811),  # pj, pj, mW, ps, ps, um^2
                (28, 64, 64): (2.258, 0.439, 3.464, 839.561, 839.561, 69543.079),
                (28, 64, 128): (2.258, 0.657, 3.459, 833.511, 833.511, 69530.745),
                (28, 64, 256): (2.258, 1.092, 3.456, 830.486, 830.486, 69524.861),

                (28, 128, 32): (4.027, 0.468, 6.897, 1.146, 1.146, 137479.428),  # pj, pj, mW, ns, ns, um^2
                (28, 128, 64): (4.027, 0.685, 6.886, 1.134, 1.134, 137484.856),
                (28, 128, 128): (4.027, 1.120, 6.881, 1.128, 1.128, 137472.156),
                (28, 128, 256): (4.453, 1.259, 6.479, 1.547, 1.547, 134323.435),

                (28, 256, 32): (7.961, 0.633, 12.908, 1.925, 1.925, 266554.126),  # pj, pj, mW, ns, ns, um^2
                (28, 256, 64): (7.957, 0.848, 12.906, 1.917, 1.917, 266236.124),
                (28, 256, 128): (7.957, 1.282, 12.893, 1.905, 1.905, 266243.448),
                (28, 256, 256): (7.957, 2.153, 12.886, 1.899, 1.899, 266226.314),

                (28, 512, 32): (14.967, 0.898, 25.715, 2.560, 2.560, 527591.787),  # pj, pj, mW, ns, ns, um^2
                (28, 512, 64): (14.964, 1.330, 25.714, 2.553, 2.553, 527279.955),
                (28, 512, 128): (14.963, 2.200, 25.700, 2.541, 2.541, 527289.103),
                (28, 512, 256): (14.963, 3.940, 25.694, 2.535, 2.535, 527267.707),


                (22, 2, 32): (0.066, 0.034, 62.983, 229.282, 229.282, 1724.060),  # pj, pj, uW, ps, ps, um^2
                (22, 2, 64): (0.094, 0.045, 61.386, 226.819, 226.819, 1722.426),
                (22, 2, 128): (0.092, 0.057, 61.045, 224.201, 224.201, 1694.848),
                (22, 2, 256): (0.174, 0.104, 61.045, 224.201, 224.201, 1694.848),

                (22, 4, 32): (0.123, 0.047, 118.202, 267.391, 267.391, 3180.006),  # pj, pj, uW, ps, ps, um^2
                (22, 4, 64): (0.123, 0.060, 117.523, 264.834, 264.834, 3178.368),
                (22, 4, 128): (0.121, 0.086, 117.181, 262.223, 262.223, 3150.366),
                (22, 4, 256): (0.230, 0.160, 117.181, 262.223, 262.223, 3150.366),

                (22, 8, 32): (0.181, 0.064, 230.754, 365.960, 365.960, 6064.926),  # pj, pj, uW, ps, ps, um^2
                (22, 8, 64): (0.234, 0.084, 220.934, 351.628, 351.628, 5935.558),
                (22, 8, 128): (0.234, 0.111, 219.859, 349.071, 349.071, 5934.023),
                (22, 8, 256): (0.231, 0.163, 219.320, 346.475, 346.475, 5882.054),

                (22, 16, 32): (0.346, 0.087, 433.591, 459.989, 459.989, 11432.357),  # pj, pj, uW, ps, ps, um^2
                (22, 16, 64): (0.346, 0.115, 431.448, 454.874, 454.874, 11427.735),
                (22, 16, 128): (0.346, 0.169, 430.373, 452.317, 452.317, 11425.527),
                (22, 16, 256): (0.343, 0.275, 429.837, 449.721, 449.721, 11372.832),

                (22, 32, 32): (0.570, 0.120, 853.083, 604.896, 604.896, 22163.819),  # pj, pj, uW, ps, ps, um^2
                (22, 32, 64): (0.570, 0.175, 850.940, 599.781, 599.781, 22158.142),
                (22, 32, 128): (0.670, 0.208, 805.831, 718.262, 718.262, 21921.581),
                (22, 32, 256): (0.670, 0.317, 804.677, 715.705, 715.705, 21919.372),

                (22, 64, 32): (1.113, 0.161, 1.602, 902.733, 902.733, 43110.524),  # pj, pj, mW, ps, ps, um^2
                (22, 64, 64): (1.113, 0.216, 1.597, 892.503, 892.503, 43113.355),
                (22, 64, 128): (1.113, 0.324, 1.595, 887.388, 887.388, 43106.728),
                (22, 64, 256): (1.112, 0.542, 1.594, 884.830, 884.830, 43103.571),

                (22, 128, 32): (1.995, 0.228, 3.178, 1.226, 1.226, 84872.504),  # pj, pj, mW, ns, ns, um^2
                (22, 128, 64): (1.994, 0.336, 3.173, 1.215, 1.215, 84868.855),
                (22, 128, 128): (1.994, 0.554, 3.171, 1.210, 1.210, 84868.015),
                (22, 128, 256): (2.192, 0.619, 2.985, 1.815, 1.815, 82924.162),

                (22, 256, 32): (3.758, 0.360, 6.327, 1.886, 1.886, 166533.978),  # pj, pj, mW, ns, ns, um^2
                (22, 256, 64): (3.944, 0.413, 5.947, 2.224, 2.224, 164360.056),
                (22, 256, 128): (3.943, 0.631, 5.941, 2.234, 2.234, 164364.578),
                (22, 256, 256): (3.943, 1.066, 5.937, 2.229, 2.229, 164354.000),

                (22, 512, 32): (7.448, 0.438, 11.850, 3.007, 3.007, 325707.174),  # pj, pj, mW, ns, ns, um^2
                (22, 512, 64): (7.446, 0.654, 11.849, 3.001, 3.001, 325514.666),
                (22, 512, 128): (7.446, 1.089, 11.843, 2.990, 2.990, 325520.313),
                (22, 512, 256): (7.446, 1.959, 2.985, 2.985, 3.007, 325507.105),
                
            }   
            
            if (self.device_tech, self.capacity, self.wordwidth) in area_dict:
                self.readenergy, self.writeenergy,self.leakagepower, self.readlatency,self.writelatency,self.area = area_dict[(self.device_tech, self.capacity, self.wordwidth)]
        
            else:
                raise IndexError("Out of range of Buf's parameter supported") 
            
            if self.area_config > 0:
                self.area = self.area_config
                
    def calculate_energy(self):
        num_buffer_used = self.channel_size*P.ACTIVATION_BIT/self.wordwidth
        
        self.read_energy = self.readenergy *self.writes * num_buffer_used *1e-12
        self.write_energy = self.writeenergy* self.writes * num_buffer_used *1e-12
        self.energy = self.read_energy + self.write_energy
        self.power = self.energy*P.FREQUENCY/self.writes
        log_print("  --energy:",self.energy)
        log_print("     --energy_write:", self.write_energy)
        log_print("     --energy_read:", self.read_energy)
        log_print("  --power:",self.power)
                
    def write(self, data):
        if len(self.buffer) < self.size:
            self.buffer.append(data)
            self.writes += 1
        return len(self.buffer)

    def read(self):
        if self.buffer:
            self.reads += 1
            return self.buffer.popleft()
        
        return None
    
    def __str__(self):
        """返回队列的字符串表示"""
        if self.show_buffer:
            return "Buffer({})".format(self.buffer)
        else: return "Buffer"
       
        

class FIFO(BufferBase):
    def __init__(self, depth, width, num:int):
        """
        Initialize a FIFO queue with a specified capacity.
        """
        self.num_FIFO = num
        super().__init__(depth, width)

        self.calculate_FIFO_area()
        self.calculate_FIFO_power()

    def _calculate_capacity(self):
        return self.depth * self.num_FIFO

    def enqueue(self, item):
        if self.is_full():
            raise OverflowError("Queue is full")
        self.queue.append(item)
        self.writes +=(1*self.depth*self.num_FIFO)
        self.total_times += (1*self.depth*self.num_FIFO)

    def dequeue(self):
       
        if self.is_empty():
            raise IndexError("Queue is empty")
        # self.reads =self.writes
        self.reads += 1
        self.total_times += 1
        return self.queue.pop(0)
    
    def calculate_FIFO_area(self):

        if self.area == 0: 
            FIFO_area_dict = {
                18:{
                    64: 9555.76
                },
                34: {
                    64: 18821.526,
                    128: 36967.79459,
                    256: 73171.07157
                },
                66: {
                    64:37327.82699,
                    128:73506.7607,
                    256:146517.132
                }
            }

        if self.depth in FIFO_area_dict:
            if self.width in FIFO_area_dict[self.depth]:
            
                self.area = FIFO_area_dict[self.depth][self.width]
            else: 
                self.area = FIFO_area_dict[self.depth][64] * self.width / 64
        else: self.area =  (578.55*self.depth +302.61 )  *self.width / 64

        
        self.area = self.area * pow((self.device_tech/40),2)* self.width/16 *self.num_FIFO

    def calculate_FIFO_power(self):

        if self.power == 0 :
            FIFO_power_dict = {
                18: { 
                    64: 9.2433e-6
                },
                34: {
                    64: 17.77e-6,
                    128: 34.81e-6,
                    256: 69.083e-6
                },
                66: {
                    64:34.737e-6,
                    128:68.73e-6,
                    256:137e-6
                }
            }

            
        if self.depth in FIFO_power_dict: 
            if self.width in FIFO_power_dict[self.depth]:
            
                self.power = FIFO_power_dict[self.depth][self.width]
                
            else: 
                self.power = FIFO_power_dict[self.depth][64] * self.width/ 64
        else: self.power = (0.531*self.depth+0.7598) *self.width/64 * 1e-6

        self.power = self.power * pow((self.device_tech/40),2) *self.num_FIFO *((P.FREQUENCY/10e6)**1.3)

    def calculate_energy(self):

        self.energy = self.power*1/P.FREQUENCY * self.writes / self.depth/self.num_FIFO
        log_print("  --power:",self.power)
        log_print("  --energy:",self.energy)


class REG_array(BufferBase):
    def __init__(self, depth:int, width:int, num ): 
        self.length = width     
        super().__init__(depth, width) 
        
        self.num_REG = num # in_channels

        # self.calculate_REG_area()
        # self.calculate_REG_power()

    def _calculate_capacity(self):
        return self.depth * self.length  # capacity of the queue


    def enqueue(self, item):
        if self.is_full():
            raise OverflowError("Queue is full")
        self.queue.append(item)
        self.writes += 1
        self.total_times += 1

    def dequeue(self):
        if self.is_empty():
            raise IndexError("Queue is empty")
        self.reads += (1*self.depth)
        self.total_times += (1*self.depth)
        return self.queue.pop(0)

    
    def calculate_REG_area(self):

        if self.area == 0: 
            FIFO_area_dict = {
                18:{
                    64: 9555.76
                },
                34: {
                    64: 18821.526,
                    128: 36967.79459,
                    256: 73171.07157
                },
                66: {
                    64:37327.82699,
                    128:73506.7607,
                    256:146517.132
                }
            }

        if self.depth in FIFO_area_dict:
            if self.width in FIFO_area_dict[self.depth]:
            
                self.area = FIFO_area_dict[self.depth][self.width]
            else: 
                self.area = FIFO_area_dict[self.depth][64] * self.width / 64
        else: self.area =  (578.55*self.depth +302.61 )  *self.width / 64

        
        self.area = self.area * pow((self.device_tech/40),2)* self.width/16 *self.num_REG

    def calculate_REG_power(self):

        if self.power == 0 :
            FIFO_power_dict = {
                18: { 
                    64: 9.2433e-6
                },
                34: {
                    64: 17.77e-6,
                    128: 34.81e-6,
                    256: 69.083e-6
                },
                66: {
                    64:34.737e-6,
                    128:68.73e-6,
                    256:137e-6
                }
            }

            
        if self.depth in FIFO_power_dict: 
            if self.width in FIFO_power_dict[self.depth]:
            
                self.power = FIFO_power_dict[self.depth][self.width]
                
            else: 
                self.power = FIFO_power_dict[self.depth][64] * self.width/ 64
        else: self.power = (0.531*self.depth+0.7598) *self.width/64 * 1e-6

        self.power = self.power * pow((self.device_tech/40),2) *self.num_REG *((P.FREQUENCY/10e6)**1.3)

    def calculate_energy(self):

        self.energy = self.power*1/P.FREQUENCY * (self.writes+ self.reads) #writes: data write, reads：data right shift or up shift
        log_print("  --power:",self.power)
        log_print("  --energy:",self.energy)


class InputBuf(BufferBase):
    def __init__(self, depth:int, width:int, num ): 
        self.length = width     
        super().__init__(depth, width) 
        
        self.num_REG = num # in_channels

        # self.calculate_REG_area()
        # self.calculate_REG_power()

    def _calculate_capacity(self):
        return self.depth * self.length  # capacity of the queue


    def enqueue(self, item):
        if self.is_full():
            raise OverflowError("Queue is full")
        self.queue.append(item)
        self.writes += 1
        self.total_times += 1

    def dequeue(self):
        if self.is_empty():
            raise IndexError("Queue is empty")
        self.reads += 1
        return self.queue.pop(0)

    
    def calculate_REG_area(self):

        if self.area == 0: 
            FIFO_area_dict = {
                18:{
                    64: 9555.76
                },
                34: {
                    64: 18821.526,
                    128: 36967.79459,
                    256: 73171.07157
                },
                66: {
                    64:37327.82699,
                    128:73506.7607,
                    256:146517.132
                }
            }

        if self.depth in FIFO_area_dict:
            if self.width in FIFO_area_dict[self.depth]:
            
                self.area = FIFO_area_dict[self.depth][self.width]
            else: 
                self.area = FIFO_area_dict[self.depth][64] * self.width / 64
        else: self.area =  (578.55*self.depth +302.61 )  *self.width / 64

        
        self.area = self.area * pow((self.device_tech/40),2)* self.width/16 *self.num_REG

    def calculate_REG_power(self):

        if self.power == 0 :
            FIFO_power_dict = {
                18: { 
                    64: 9.2433e-6
                },
                34: {
                    64: 17.77e-6,
                    128: 34.81e-6,
                    256: 69.083e-6
                },
                66: {
                    64:34.737e-6,
                    128:68.73e-6,
                    256:137e-6
                }
            }

            
        if self.depth in FIFO_power_dict: 
            if self.width in FIFO_power_dict[self.depth]:
            
                self.power = FIFO_power_dict[self.depth][self.width]
                
            else: 
                self.power = FIFO_power_dict[self.depth][64] * self.width/ 64
        else: self.power = (0.531*self.depth+0.7598) *self.width/64 * 1e-6

        self.power = self.power * pow((self.device_tech/40),2) *self.num_REG *((P.FREQUENCY/10e6)**1.3)

    def calculate_energy(self):

        self.energy = self.power*1/P.FREQUENCY * (self.writes+ self.reads) #writes: data write, reads：data right shift or up shift
        log_print("  --power:",self.power)
        log_print("  --energy:",self.energy)

        
if __name__ == "__main__":

    HwConfig_path = './config.ini'
    Hw_config = configparser.ConfigParser()
    Hw_config.read(HwConfig_path, encoding = 'UTF-8')

    buf = Global_buffer(Hw_config, buffer_level= 1)
    log_print(buf.area)