import numpy as np
from hutch_python.utils import safe_load
from pcdsdevices import analog_signals
from xpp.db import xpp_attenuator as att
from xpp.db import xpp_jj_3 as xpp_jj_3
from xpp.db import xpp_jj_2 as xpp_jj_2
from xpp.db import daq, pp
from time import sleep
from ophyd import EpicsSignal
from ophyd import EpicsSignalRO
from ophyd import EpicsMotor

from pcdsdevices.beam_stats import BeamEnergyRequest, BeamEnergyRequestACRWait

from pcdsdevices import analog_signals

# import pcdsdevices.analog_signals as analog_signals

SET_WAIT = 2


class SettleSignal(EpicsSignal):
    def __init__(self, *args, settle_time=None, **kwargs):
        self._settle_time = settle_time
        super().__init__(*args, **kwargs)

    def set(self, *args, **kwargs):
        return super().set(*args, settle_time=self._settle_time, **kwargs)


class User():
    def __init__(self):
        self.energy_set = SettleSignal('XPP:USER:MCC:EPHOT:SET1',
                                       name='energy_set',
                                       settle_time=SET_WAIT)
        self.energy_ref = SettleSignal('XPP:USER:MCC:EPHOT:REF1',
                                       name='energy_ref')

        self.acr_energy = BeamEnergyRequestACRWait(name='acr_energy',
                                                   prefix='XPP',
                                                   acr_status_suffix='AO805')

        # with safe_load('analog_out'):
        self.aio = analog_signals.Acromag(name='xpp_aio', prefix='XPP:USR')
        self.dl = EpicsMotor(prefix='XPP:ENS:01:m0', name='aero')

    def miniSD_clear_beam(self):
        self.t1x.umv(-9.3)
        self.t2x.umv(36.5)
        self.t3x.umv(7.0)
        self.t6x.umv(5.0)
        self.d2x.umv(0)
        self.d3x.umv(-1)
        self.d4x.umv(0)

    def miniSD_diods_remove_forData(self):
        self.d2x.umv(0)
        self.d3x.umv(20)
        self.d4x.umv(40)

    def miniSD_diods_incert_forAlign(self):
        self.d2x.umv(-22)
        self.d3x.umv(0)
        self.d4x.umv(22)

    def show_CC(self):
        self.aio.ao1_2.set(0)
        self.aio.ao1_3.set(5)

    def show_VCC(self):
        self.aio.ao1_2.set(5)
        self.aio.ao1_3.set(0)

    def show_both(self):
        self.aio.ao1_2.set(5)
        self.aio.ao1_3.set(5)

    def show_neither(self):
        self.aio.ao1_2.set(0)
        self.aio.ao1_3.set(0)

    def move_unfocused(self):
        self.crl_x.umv_out()
        self.crl_y.umv_out()
        att(1)
        xpp_jj_3.hg(1)

    def move_focused(self):
        att(0.01)
        self.crl_x.umv_in()
        self.crl_y.umv_in()
        xpp_jj_3.hg(0.25)

    def move_delay_offset(self, offset):
        self.t2x.umvr(offset)
        self.t3x.umvr(offset)

    def move_unfocused_beam(self, offset):
        self.t2x.umvr(offset)
        self.t3x.umvr(-offset)

    def move_focused_beam(self, offset):
        if offset > 4e-4:
            print("this step seems to be rather large")
        self.t3th.umvr(offset)
        self.t5th.umvr(-offset)

    def dumbSnake(self, xStart, xEnd, yDelta, nRoundTrips, sweepTime):
        """
        simple rastering for running at 120Hz with shutter open/close before
        and after motion stop.

        Need some testing how to deal with intermittent motion errors.
        """
        self.sam_x.umv(xStart)
        daq.connect()
        daq.begin()
        sleep(2)
        print('Reached horizontal start position')
        # looping through n round trips
        for i in range(nRoundTrips):
            try:
                print('starting round trip %d' % (i + 1))
                self.sam_x.mv(xEnd)
                sleep(0.1)
                pp.open()
                sleep(sweepTime)
                pp.close()
                self.sam_x.wait()
                self.sam_y.umvr(yDelta)
                sleep(1.2)  # orignal was 1
                self.sam_x.mv(xStart)
                sleep(0.1)
                pp.open()
                sleep(sweepTime)
                pp.close()
                self.sam_x.wait()
                self.sam_y.umvr(yDelta)
                print('ypos', self.sam_y.wm())
                sleep(0.5)  # original was 1
            except:
                print('round trip %d didn not end happily' % i)
        daq.end_run()
        daq.disconnect()
