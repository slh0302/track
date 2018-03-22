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
            print('    recall %.4f, prec %.4f, mAP %.4f' % (np.average(rec), np.average(prec), ap) )

        print('Mean AP = {:.4f}'.format(np.mean(aps)))

    def evaluate_recall(self, candidate_boxes=None, thresholds=None,
                        area='all', limit=None):
        """Evaluate detection proposal recall metrics.

        Returns:
            results: dictionary of results with keys
                'ar': average recall
                'recalls': vector recalls at each IoU overlap threshold
                'thresholds': vector of IoU overlap thresholds
                'gt_overlaps': vector of all ground-truth overlaps
        """
        # Record max overlap value for each gt box
        # Return vector of overlap values
        areas = {'all': 0, 'small': 1, 'medium': 2, 'large': 3,
                 '96-128': 4, '128-256': 5, '256-512': 6, '512-inf': 7}
        area_ranges = [[0 ** 2, 1e5 ** 2],  # all
                       [0 ** 2, 32 ** 2],  # small
                       [32 ** 2, 96 ** 2],  # medium
                       [96 ** 2, 1e5 ** 2],  # large
                       [96 ** 2, 128 ** 2],  # 96-128
                       [128 ** 2, 256 ** 2],  # 128-256
                       [256 ** 2, 512 ** 2],  # 256-512
                       [512 ** 2, 1e5 ** 2],  # 512-inf
                       ]
        assert areas.has_key(area), 'unknown area range: {}'.format(area)
        area_range = area_ranges[areas[area]]
        gt_overlaps = np.zeros(0)
        num_pos = 0
        for i in range(self.num_images):
            # Checking for max_overlaps == 1 avoids including crowd annotations
            # (...pretty hacking :/)
            max_gt_overlaps = self.roidb[i]['gt_overlaps'].toarray().max(axis=1)
            gt_inds = np.where((self.roidb[i]['gt_classes'] > 0) &
                               (max_gt_overlaps == 1))[0]
            gt_boxes = self.roidb[i]['boxes'][gt_inds, :]
            gt_areas = self.roidb[i]['seg_areas'][gt_inds]
            valid_gt_inds = np.where((gt_areas >= area_range[0]) &
                                     (gt_areas <= area_range[1]))[0]
            gt_boxes = gt_boxes[valid_gt_inds, :]
            num_pos += len(valid_gt_inds)

            if candidate_boxes is None:
                # If candidate_boxes is not supplied, the default is to use the
                # non-ground-truth boxes from this roidb
                non_gt_inds = np.where(self.roidb[i]['gt_classes'] == 0)[0]
                boxes = self.roidb[i]['boxes'][non_gt_inds, :]
            else:
                boxes = candidate_boxes[i]
            if boxes.shape[0] == 0:
                continue
            if limit is not None and boxes.shape[0] > limit:
                boxes = boxes[:limit, :]

            overlaps = bbox_overlaps(boxes.astype(np.float),
                                     gt_boxes.astype(np.float))

            _gt_overlaps = np.zeros((gt_boxes.shape[0]))
            for j in range(gt_boxes.shape[0]):
                # find which proposal box maximally covers each gt box
                argmax_overlaps = overlaps.argmax(axis=0)
                # and get the iou amount of coverage for each gt box
                max_overlaps = overlaps.max(axis=0)
                # find which gt box is 'best' covered (i.e. 'best' = most iou)
                gt_ind = max_overlaps.argmax()
                gt_ovr = max_overlaps.max()
                assert (gt_ovr >= 0)
                # find the proposal box that covers the best covered gt box
                box_ind = argmax_overlaps[gt_ind]
                # record the iou coverage of this gt box
                _gt_overlaps[j] = overlaps[box_ind, gt_ind]
                assert (_gt_overlaps[j] == gt_ovr)
                # mark the proposal box and the gt box as used
                overlaps[box_ind, :] = -1
                overlaps[:, gt_ind] = -1
            # append recorded iou coverage level
            gt_overlaps = np.hstack((gt_overlaps, _gt_overlaps))

        gt_overlaps = np.sort(gt_overlaps)
        if thresholds is None:
            step = 0.05
            thresholds = np.arange(0.5, 0.95 + 1e-5, step)
        recalls = np.zeros_like(thresholds)
        # compute recall for each iou threshold
        for i, t in enumerate(thresholds):
            recalls[i] = (gt_overlaps >= t).sum() / float(num_pos)
        # ar = 2 * np.trapz(recalls, thresholds)
        ar = recalls.mean()
        return {'ar': ar, 'recalls': recalls, 'thresholds': thresholds,
                'gt_overlaps': gt_overlaps}