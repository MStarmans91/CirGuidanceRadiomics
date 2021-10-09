import WORC
import os
import glob


def editconfig(config, model='1a'):
    # Use Segmentix script to adjust segmentations
    config['General']['Segmentix'] = 'True'

    # No image normalization for CT
    config['Preprocessing']['Normalize'] = 'False'
    config['Preprocessing']['Normalize_ROI'] = 'True'

    # For naming of features
    config['ImageFeatures']['image_type'] = 'CT'

    # Turn this to True if you want to use the features from ExampleData\semantics.csv
    config['SelectFeatGroup']['semantic_features'] = 'False'

    # Always use resampling, as the dataset is imbalanced
    config['Resampling']['Use'] = '1.00'

    # Label to predict
    config['Labels']['modus'] = 'singlelabel'
    config['Labels']['label_names'] = 'pNStage'

    if model in ['1a', '2a']:
        # Keep all blobs and do not fill holes to fill holes between blobs
        config['Segmentix']['N_blobs'] = '0'
        config['Segmentix']['fillholes'] = 'False'

    elif model in ['1b', '2b']:
        # Combine features of same patient
        config['FeatPreProcess']['Combine'] = 'True'
        config['FeatPreProcess']['Combine_method'] = 'mean'

    elif model == '3a':
        # Only keep the five largest blobs
        config['Segmentix']['N_blobs'] = '5'
        config['Segmentix']['fillholes'] = 'False'

    elif model == '3b':
        # Only keep the five largest blobs
        config['Segmentix']['N_blobs'] = '5'
        config['Segmentix']['fillholes'] = 'False'

        # Combine features of same patient
        config['FeatPreProcess']['Combine'] = 'True'
        config['FeatPreProcess']['Combine_method'] = 'mean'
    else:
        raise KeyError(f'Name {model} is invalid as model!')

    return config


# Determine which model you want to fit: 1a, 1b, 2a, 2b, 3a, or 3b, see paper
model = '1a'
name = f'WORC_CirGuidance_{model}'

# Inputs
current_path = os.path.dirname(os.path.abspath(__file__))
label_file = os.path.join(current_path, 'ExampleData', 'pinfo.csv')
semantics_file = os.path.join(current_path, 'ExampleData', 'sem.csv')
config = os.path.join(current_path, 'ExampleData', 'config.ini')

# We advise you to provide WORC with the raw images and segmentations. In
# this example,  we will supply the extracted features directly.
feature_files = glob.glob(os.path.join(current_path, 'ExampleData', 'example_features_predict_CirGuidanceRadiomics-*.hdf5'))
feature_files.sort()

# As we only have a single feature file, we will repeat it to mimick
# having multiple. We do this in a dictionary, in which the keys
# correspond to the "patient" names also used in the label and semantics files
patient_names = ['CirGuidanceRadiomics-' + str(i).zfill(3) for i in range(0, 10)]
features = {k: v for k, v in zip(patient_names, feature_files)}

# Create the WORC network
network = WORC.WORC(name)

# Instead of supplying the .ini file to the network, we will create
# the config object for you directly from WORC,
# so you can interact with it if you want.
# Altough it is a configparser object, it works similar as a dictionary
config = network.defaultconfig()

# Change config settings for this study and the respective model
config = editconfig(config, model)

# NOTE: Since we now only use 10 "patients" in this example, we do not use resampling.
# Do not do this for the full experiment.
config['Resampling']['Use'] = '0.0'

# Append the sources to be used. When using raw images and segmentations, use the
# images_train and segmentations_train instead of features_train object.
network.features_train.append(features)
network.labels_train.append(label_file)
network.semantics_train.append(semantics_file)
network.configs.append(config)

# Build, set, and execture the network
network.build()
network.set()
network.execute()

# NOTE: if you want extensive evaluation including ROC curves, statistical
# testing of features, add ``network.add_evaluation('pNStage')'' after
# network.build().
