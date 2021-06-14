import os
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt

AUTOTUNE = tf.data.experimental.AUTOTUNE


def plot_history(history):
	print(history.history)

	plt.figure(figsize=(8, 8))
	plt.subplot(1, 2, 1)
	plt.plot(history.history["sparse_categorical_accuracy"], label="Training Accuracy")
	plt.plot(history.history["val_sparse_categorical_accuracy"], label="Validation Accuracy")
	plt.title("Model accuracy")
	plt.xlabel("Epoch")
	plt.legend(loc="lower right")

	plt.subplot(1, 2, 2)
	plt.plot(history.history["loss"], label="Training Loss")
	plt.plot(history.history["val_loss"], label="Validation Loss")
	plt.xlabel("Epoch")
	plt.legend(loc="lower left")
	plt.title('Training and Validation Loss')

	plt.show()

def parse_tfrecord_fn(example):
	feature_description = {
		"image": tf.io.FixedLenFeature([], tf.string),
		"path": tf.io.FixedLenFeature([], tf.string),
		"class_id": tf.io.FixedLenFeature([], tf.int64),
	}
	example = tf.io.parse_single_example(example, feature_description)
	example["image"] = tf.io.decode_png(example["image"], channels=3)
	return example

def prepare_sample(features):
	image = tf.image.resize(features["image"], size=(256, 256))
	return image, features["class_id"]

def get_dataset(filenames, batch_size):
	dataset = (
		tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
		.map(parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)
		.map(prepare_sample, num_parallel_calls=AUTOTUNE)
		.batch(batch_size)
		.prefetch(AUTOTUNE)
	)
	return dataset

def build_efficientnetBX(is_pretrained, num_model):
	"""Build the EfficientnetBX model, X being 0 to 7"""
	# reject requested model if number's not in range(8)
	if num_model not in range(8):
		raise ValueError("Model number requested not in range 0 to 7!")
	# pick the requested model
	if num_model == 0:
		EfficientNetBX = tf.keras.applications.EfficientNetB0
	elif num_model == 1:
		EfficientNetBX = tf.keras.applications.EfficientNetB1
	elif num_model == 2:
		EfficientNetBX = tf.keras.applications.EfficientNetB2
	elif num_model == 3:
		EfficientNetBX = tf.keras.applications.EfficientNetB3
	elif num_model == 4:
		EfficientNetBX = tf.keras.applications.EfficientNetB4
	elif num_model == 5:
		EfficientNetBX = tf.keras.applications.EfficientNetB5
	elif num_model == 6:
		EfficientNetBX = tf.keras.applications.EfficientNetB6
	elif num_model == 7:
		EfficientNetBX = tf.keras.applications.EfficientNetB7

	classes = 25
	weights = "imagenet" if is_pretrained else None
	include_top = False if is_pretrained else True

	# define image augmentation layers
	image_augmentation = tf.keras.models.Sequential(
		[
			tf.keras.layers.experimental.preprocessing.RandomRotation(factor=0.15),
			tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
			tf.keras.layers.experimental.preprocessing.RandomFlip(),
			tf.keras.layers.experimental.preprocessing.RandomContrast(factor=0.1)
		],
		name="image_augmentation",
	)

	input_tensor = tf.keras.layers.Input(shape=(256, 256, 3))
	x = image_augmentation(input_tensor)
	model = EfficientNetBX(
		input_tensor=x, weights=weights, 
		include_top=include_top, classes=classes
	)
	if not is_pretrained:
		model.compile(
			optimizer=tf.keras.optimizers.Adam(),
			loss=tf.keras.losses.SparseCategoricalCrossentropy(),
			metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
		)
		return model

	# freeze the pretrained weights
	model.trainable = False

	# rebuild top
	x = tf.keras.layers.GlobalAveragePooling2D()(model.output)
	outputs = tf.keras.layers.Dense(classes, activation="softmax")(x)
	model = tf.keras.Model(input_tensor, outputs)

	# instantiate model and compile
	model.compile(
		optimizer=tf.keras.optimizers.Adam(),
		loss=tf.keras.losses.SparseCategoricalCrossentropy(),
		metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
	)
	return model

def unfreeze_model(model):
	# unfreeze all layers except batch normalization
	for layer in model.layers:
		if not isinstance(layer, tf.layers.BatchNormalization):
			layer.trainable = True
	# recompile the model
	model.compile(
		optimizer=tf.keras.optimizers.Adam(),
		loss=tf.keras.losses.SparseCategoricalCrossentropy(),
		metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
	)



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--pretrained",
		help="Train the model with pretrained imagenet weights.", 
		default=False,
		action="store_true"
	)
	parser.add_argument(
		"-B", "--B", 
		type=int,
		help="B number of the model EfficientNetBX.", 
		default=0,
	)
	parser.add_argument(
		"-ckpt", "--load_checkpoint", 
		type=str,
		help="Load checkpoint path from an earlier run.", 
	)
	parser.add_argument(
		"-mdl", "--load_model", 
		type=str,
		help="Load model path from an earlier run.", 
	)
	args = parser.parse_args()

	tfrecords_dir = "/media/david/TOSHIBA EXT/DavidINLOC/tfmdbs/freiburg_groceries_dataset/tfrecords"
	train_dir = os.path.join(tfrecords_dir, "train")
	val_dir = os.path.join(tfrecords_dir, "val")
	
	# list of tfrecords for training and validation
	train_filenames = tf.io.gfile.glob(train_dir + "/*.tfrec")
	val_filenames = tf.io.gfile.glob(val_dir + "/*.tfrec")

	# build the model and load checkpoint if requested, and train from there
	model = build_efficientnetBX(args.pretrained, args.B)
	if args.load_checkpoint:
		model.load_weights(args.load_checkpoint)
	
	# laod an entire model with its checkpoints, and train from there
	if args.load_model:
		model = tf.keras.models.load_model(args.load_model)
		model.trainable = True

	# Train
	batch_size = 64
	epochs = 50

	train_ds = get_dataset(train_filenames, batch_size)
	val_ds = get_dataset(val_filenames, batch_size)

	len_train_ds = sum(1 for _ in train_ds)
	len_val_ds = sum(1 for _ in val_ds)
	steps_per_epoch = 2 * len_train_ds # multiplier: due to the augmentation
	validation_steps = 1 * len_val_ds

	callbacks = [
		tf.keras.callbacks.ModelCheckpoint(
			filepath="EfficientB0_freiburg.h5", 
			save_weights_only=False,
			save_best_only=True,
			monitor="val_loss",
			verbose=1),
	]
	model.summary()
	history = model.fit(
		x=train_ds.repeat(),
		steps_per_epoch=steps_per_epoch,
		validation_data=val_ds.repeat(),
		validation_steps=validation_steps,
		epochs=epochs,
		callbacks=callbacks,
		verbose=1,
	)

	plot_history(history)