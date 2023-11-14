import inspect

class AUG_TYPE:
    RAIN = 'rain'
    BLUR = 'blur'
    CLOUD = 'cloud'
    SWITCH_CHANNEL = 'switch_channel'
    RESIZE_AND_CROP = 'resize_and_crop'
    FLIP_H = 'flip_h'
    FLIP_V = 'flip_v'
    SWITCH_UP_AND_DOWN = 'switch_up_and_down'
    SWITCH_LEFT_AND_RIGHT = 'switch_left_and_right'

    @staticmethod
    def list_available_augmentation_methods():
        for i in inspect.getmembers(AUG_TYPE):
            # to remove private and protected
            # functions
            if not i[0].startswith('_') and not inspect.isfunction(i[1]):
                print(i)

