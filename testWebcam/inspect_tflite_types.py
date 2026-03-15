import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="fall_multitask_f32_builtins.tflite")
interpreter.allocate_tensors()

for t in interpreter.get_tensor_details():
    print(t['name'], t['dtype'])