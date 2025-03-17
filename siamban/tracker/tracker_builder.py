from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from siamban.core.config import cfg
from siamban.tracker.siamban_tracker import SiamBANTracker
from siamban.tracker.siamban_tracker_v import SiamBANTracker as SiamBANTrackerV
from siamban.tracker.siamban_tracker_v3d import SiamBANTracker as SiamBANTrackerV3d
TRACKS = {
          'SiamBANTracker': SiamBANTracker,
          'SiamBANTrackerV': SiamBANTrackerV,
          'SiamBANTrackerV3d': SiamBANTrackerV3d,
         }


def build_tracker(model):
    return TRACKS[cfg.TRACK.TYPE](model)
