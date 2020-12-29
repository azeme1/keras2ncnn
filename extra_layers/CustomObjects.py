from extra_layers.Clip import Clip
from extra_layers.OutputSplit import OutputSplit
from extra_layers.Padding import ReflectPadding2D
from extra_layers.BinaryOp import Subtract
from extra_layers.UnaryOp import Sqrt


extra_custom_objects = {'OutputSplit': OutputSplit, 'Clip': Clip, 'ReflectPadding2D': ReflectPadding2D,
                        'Subtract': Subtract, 'Sqrt': Sqrt}


