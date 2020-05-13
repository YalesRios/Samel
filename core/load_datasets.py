import tensorflow as tf
import keras

def load_data(dataset):
    if (dataset == "MNIST"):
        data = tf.keras.datasets.mnist

        (trainX, trainY),(testX, testY) = data.load_data()

        trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
        testX = testX.reshape((testX.shape[0], 28, 28, 1))

        trainY = keras.utils.to_categorical(trainY, 10)
        testY = keras.utils.to_categorical(testY, 10)
        
        trainX = trainX.astype("float32") / 255.0
        testX = testX.astype("float32") / 255.0

        return (trainX, trainY),(testX, testY)

    elif (dataset == "NotMNIST"):
        folders = ['../datasets/notMNIST_small/A',
                   '../datasets/notMNIST_small/B',
                   '../datasets/notMNIST_small/C',
                   '../datasets/notMNIST_small/D',
                   '../datasets/notMNIST_small/E',
                   '../datasets/notMNIST_small/F',
                   '../datasets/notMNIST_small/G',
                   '../datasets/notMNIST_small/H',
                   '../datasets/notMNIST_small/I',
                   '../datasets/notMNIST_small/J']
        
        image_size = 28  # Pixel width and height.
        pixel_depth = 255.0  # Number of levels per pixel.
        
        # Source: https://www.ritchieng.com/machine-learning/deep-learning/tensorflow/notmnist/
        def load_letter(folder, min_num_images):
          """Load the data for a single letter label."""
          image_files = os.listdir(folder)
          dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                                 dtype=np.float32)
          print(folder)
          num_images = 0
          for image in image_files:
            image_file = os.path.join(folder, image)
            try:
              image_data = (imageio.imread(image_file).astype("float32") - 
                            pixel_depth / 2) / pixel_depth
              if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
              dataset[num_images, :, :] = image_data
              num_images = num_images + 1
            except (IOError, ValueError) as e:
              print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
            
          dataset = dataset[0:num_images, :, :]
          if num_images < min_num_images:
            raise Exception('Many fewer images than expected: %d < %d' %
                            (num_images, min_num_images))
            
          return dataset
        
        letters = []
        for folder in folders:
            letters.extend(load_letter(folder,1))
        letters = np.asarray(letters)
        
        letters = letters.reshape((letters.shape[0],28,28,1))

        return letters
        
    elif (dataset == "SVHN"):
        tf.enable_eager_execution()
        
        data = tfds.load("svhn_cropped", split = ['train','test'])
        
        tf.executing_eagerly()
        
        train, test = data
        
        train = np.array(list(tfds.as_numpy(train)))
        test = np.array(list(tfds.as_numpy(test)))
        
        trainX = np.array([inst['image'] for inst in train])
        trainY = np.array([inst['label'] for inst in train])
        testX = np.array([inst['image'] for inst in test])
        testY = np.array([inst['label'] for inst in test])
        
        trainY = keras.utils.to_categorical(trainY, 10)
        testY = keras.utils.to_categorical(testY, 10)
        
        trainX = trainX.astype("float32") / 255.0
        testX = testX.astype("float32") / 255.0
        
        tf.compat.v1.disable_eager_execution()

        return (trainX, trainY),(testX, testY)

    elif (dataset == "CIFAR10"):
        data_cifar = tf.keras.datasets.cifar10.load_data()

        (trainX, trainY),(testX, testY) = data_cifar

        trainY = keras.utils.to_categorical(trainY, 10)
        testY = keras.utils.to_categorical(testY, 10)
        
        trainX = trainX.astype("float32") / 255.0
        testX = testX.astype("float32") / 255.0

        return (trainX, trainY),(testX, testY)
