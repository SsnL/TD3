from matplotlib.animation import FFMpegWriter


class FasterFFMpegWriter(FFMpegWriter):
    '''FFMpeg-pipe writer bypassing figure.savefig.'''
    def __init__(self, **kwargs):
        '''Initialize the Writer object and sets the default frame_format.'''
        super().__init__(**kwargs)
        self.frame_format = 'rgb24'

    def grab_frame(self, **savefig_kwargs):
        assert len(savefig_kwargs) == 0
        '''Grab the image information from the figure and save as a movie frame.

        Doesn't use savefig to be faster: savefig_kwargs will be ignored.
        '''
        # re-adjust the figure size and dpi in case it has been changed by the
        # user.  We must ensure that every frame is the same size or
        # the movie will not save correctly.
        self.fig.set_size_inches(self._w, self._h)
        self.fig.set_dpi(self.dpi)
        # Draw and save the frame as an argb string to the pipe sink
        self.fig.canvas.draw()
        self._frame_sink().write(self.fig.canvas.tostring_rgb())
