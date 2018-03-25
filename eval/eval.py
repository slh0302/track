from eval.voc_eval import voc_eval, voc_eval_one
import numpy as np
import os
import pickle

class eval():

    def __init__(self, classes, devkit_path, det_id, clean_up=False):
        self._classes = classes
        self._devkit_path = devkit_path
        self._det_ids = det_id
        self._cleanup = clean_up

    def evaluate_detections(self, all_boxes, output_dir, ansname='TestAns.pkl'):
        self._write_voc_results_file(all_boxes)
        self._do_python_eval(output_dir, ansname=ansname)

        if self._cleanup:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                os.remove(filename)

    def evaluate_SingleDetections(self, all_boxes, output_dir, ansname='TestAns.pkl'):
        self._write_voc_results_file(all_boxes)
        self._do_python_eval(output_dir, ansname=ansname, oneClass=True)

        if self._cleanup:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                os.remove(filename)

    def _get_voc_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._det_ids + '_det_' + '_{:s}.txt'
        path = os.path.join(
            self._devkit_path,
            'results',
            filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls in self._classes:
            if cls == '__background__':
                continue
            print('Writing {} VOC results file'.format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            type_boxes = all_boxes[cls]
            with open(filename, 'wt') as f:
                for im_ind in type_boxes:
                    dets = type_boxes[im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(len(dets)):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(im_ind, dets[k][-1],
                                       dets[k][0] , dets[k][1],
                                       dets[k][2], dets[k][3]))

    def _do_python_eval(self, output_dir='output', ansname='TestAns.pkl', oneClass=False):
        cachedir = os.path.join(self._devkit_path, ansname)
        aps = []

        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            if not oneClass:
                rec, prec, ap = voc_eval(
                    filename, cls, cachedir, ovthresh=0.5,
                    use_07_metric=False)
            else:
                rec, prec, ap = voc_eval_one(
                    filename, cls, cachedir, ovthresh=0.5,
                    use_07_metric=False)
            aps += [ap]
            print('class {}'.format(cls))
            print('    recall %.4f, prec %.4f, mAP %.4f' % (rec, prec, ap) )

        print('Mean AP = {:.4f}'.format(np.mean(aps)))

