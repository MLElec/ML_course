import ml_utils.road_seg as rs
import ml_utils.model as model
import ml_utils.data_augmentation as d_aug
import ml_utils.postprocess as pproc
import argparse
import os
import numpy as np
# m = model.Model(model_type='cnn_bn')


def parse_args():
    parser = argparse.ArgumentParser(description='Run model for road classification')
    parser.add_argument('--model', dest='model', default='', help='sum the integers (default: find the max)')
    args = parser.parse_args()

    return args

if __name__ == "__main__":

    args = parse_args()

    print('\nSet paths')
    path_data = 'data'
    path_train_dir = os.path.join(path_data, 'training')
    path_test = os.path.join(path_data, 'test_set_images')
    path_models = 'model'

    if args.model == '':
        print('*********************************\n'
              '************ WARNING ************\n'
              '*********************************\n'
              'Run full training. It can take up to 18 hours on gpu (Telsa K80) !!! '
              'We recommend to use pre-trained model !\n')
    else:
        print('Called with model file: {}'.format(args.model))

    print('\nBuilding model ....')
    m = model.Model(model_type='cnn_bn')

    if args.model == '':
        m.train_model(path_train_dir, n_epoch=150, display_epoch=5, nmax=10, n_aug=400, n_worst=50, ratio=1.0)
        m.plot_stats()
        file_save = m.save_path_model

    else:
        file_save = args.model

    print('\nLoad test set ....')
    m._get_base_sets(ratio=1.0)
    test_img = rs.load_test_set(path_test)
    test_img_norm, _, _ = rs.normalize_data(test_img, mode='all', mean_ref=m.mean, std_ref=m.std)
    print('\nApply model on test set ....')
    y_pred_test = m.apply_model(test_img_norm, file_save, nmax=2)
    print('\nProcess images')
    y_pred_test_proc = pproc.process_all(y_pred_test, size_image=608)
    print('\nDisplay predictions')
    im_proc = np.reshape(y_pred_test_proc, (-1, 608, 608))
    rs.display_predictions(y_pred_test, test_img, im_proc, n_display=10)
    print('\nWrite submission file')
    rs.create_submission(im_proc, 'submission_final.csv')
