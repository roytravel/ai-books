import tensorflow as tf
from tensorflow.python.client import device_lib

# Is GPU available in TensorFlow?
print (device_lib.list_local_devices())
print (tf.test.is_built_with_cuda())
print (tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))