class args:
    from . import const as _c

    # names of all parameters

    varlist = [ ['nr',        int],  # noqa
                ['nt',        int],  # noqa
                ['tmax',    float],  # noqa
                ['alpha',   float],  # noqa
                ['d2g',     float],  # noqa
                ['mstar',   float],  # noqa
                ['tstar',   float],  # noqa
                ['rstar',   float],  # noqa
                ['rc',      float],  # noqa
                ['rt',      float],  # noqa
                ['r0',      float],  # noqa
                ['r1',      float],  # noqa
                ['mdisk',   float],  # noqa
                ['rhos',    float],  # noqa
                ['vfrag',   float],  # noqa
                ['vfrag_in',float],  # noqa
                ['vfrag_out',float],  # noqa
                ['a0',      float],  # noqa
                ['gamma',   float],  # noqa
                ['edrift',  float],  # noqa
                ['estick',  float],  # noqa
                ['T',       float],  # noqa
                ['gasevol',  bool],  # noqa
                ['tempevol', bool],  # noqa
                ['starevol', bool],  # noqa
                ['dir',      str],  # noqa
                ['alpha_gas',float],
                ['gap_depth',float],
                ['gap_width',float],
                ['gap_loc',float],
                ['M_seed',float],
                ['M_iso',float]
            ]

    # set default values

    nr      = 200               # noqa
    nt      = 100               # noqa
    na      = 150               # noqa
    tmax    = 1e6*_c.year       # noqa
    alpha   = 1e-3              # noqa
    d2g     = 1e-2              # noqa
    mstar   = 0.7*_c.M_sun      # noqa
    tstar   = 4010.             # noqa
    rstar   = 1.806*_c.R_sun    # noqa
    rc      = 200*_c.AU         # noqa
    rt      = 1e6*_c.AU         # noqa
    r0      = 0.05*_c.AU        # noqa
    r1      = 3e3*_c.AU         # noqa
    mdisk   = 0.1*mstar         # noqa
    rhos    = 1.156             # noqa
    vfrag   = None              # noqa
    vfrag_in= 1e2               # noqa
    vfrag_out= 1e3              # noqa
    a0      = 1e-5              # noqa
    gamma   = 1.0               # noqa
    edrift  = 1.0               # noqa
    estick  = 1.0               # noqa

    gasevol  = True   # noqa
    tempevol = False  # noqa
    starevol = False  # noqa
    T        = None   # noqa
    dir      = 'data' # noqa
    
    alpha_gas = 1e-3
    gap_depth = None
    gap_width = None
    gap_loc   = None
    
    M_seed    = None
    M_iso     = None

    def __init__(self, **kwargs):
        """
        Initialize arguments. Simulation parameters can be given as keywords.
        To list all parameters, call `print_args()`. All quantities need to be
        given in CGS units, even though they might be printed by `print_args`
        using other units.
        """
        import warnings
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                warnings.warn("No such argument")

    def __str__(self):
        """
        String representation of arguments
        """
        from .const import year, M_sun, R_sun, AU, M_earth
        from numbers import Number

        s = ''
        s += 48 * '-' + '\n'

        conversion = {
            'nr':      [1,            ''],              # noqa
            'nt':      [1,            ''],              # noqa
            'tmax':    [1/year,       'years'],         # noqa
            'd2g':     [1,            ''],              # noqa
            'mstar':   [1/M_sun,      'solar masses'],  # noqa
            'tstar':   [1,            'K'],             # noqa
            'rstar':   [1/R_sun,      'R_sun'],         # noqa
            'rc':      [1/AU,         'AU'],            # noqa
            'rt':      [1/AU,         'AU'],            # noqa
            'r0':      [1/AU,         'AU'],            # noqa
            'r1':      [1/AU,         'AU'],            # noqa
            'mdisk':   [1/self.mstar, 'M_star'],        # noqa
            'rhos':    [1,            'g/cm^3'],        # noqa
            'a0':      [1,            'cm'],            # noqa
            'gamma':   [1,            ''],              # noqa
            'edrift':  [1,            ''],              # noqa
            'estick':  [1,            '']               # noqa
            }

        for n, conv_unit in conversion.items():
            conv, unit = conv_unit
            value = getattr(self, n)
            isarray = hasattr(value, '__len__')
            if isarray:
                s += n.ljust(17) + ' = ' + '[{:3.2g} ... {:3.2g}]'.format(conv * value[0], conv * value[-1]).rjust(15) + ' ' + unit + '\n'
            else:
                s += n.ljust(17) + ' = ' + '{:3.2g}'.format(conv * value).rjust(15) + ' ' + unit + '\n'

        # print other arguments

        s += 'Output directory '.ljust(17) + ' = ' + self.dir.rjust(15) + '\n'
        s += 'Gas         evol.'.ljust(17) + ' = ' + (self.gasevol * 'on' + (not self.gasevol) * 'off').rjust(15) + '\n'
        s += 'Temperature evol.'.ljust(17) + ' = ' + (self.tempevol * 'on' + (not self.tempevol) * 'off').rjust(15) + '\n'
        s += 'Stellar     evol.'.ljust(17) + ' = ' + (self.starevol * 'on' + (not self.starevol) * 'off').rjust(15) + '\n'

        # print temperature

        if self.T is None:
            s += 'Temperature'.ljust(17) + ' = ' + 'None'.rjust(15)
        elif hasattr(self.T, '__call__'):
            s += 'Temperature'.ljust(17) + ' = ' + 'function'.rjust(15)
        else:
            s += 'Temperature'.ljust(17) + ' = ' + '{}'.format(self.T).rjust(15)
        s += '\n'

        # print alpha

        if isinstance(self.alpha, Number):
            s += 'alpha'.ljust(17) + ' = ' + '{}'.format(self.alpha).rjust(15)
        elif hasattr(self.alpha, '__call__'):
            s += 'alpha'.ljust(17) + ' = ' + 'function'.rjust(15)
        else:
            s += 'alpha'.ljust(17) + ' = ' + '{}'.format(self.alpha).rjust(15)

        s += '\n'
        
        # print alpha_gas

        if isinstance(self.alpha_gas, Number):
            s += 'alpha_gas'.ljust(17) + ' = ' + '{}'.format(self.alpha_gas).rjust(15)
        elif hasattr(self.alpha_gas, '__call__'):
            s += 'alpha_gas'.ljust(17) + ' = ' + 'function'.rjust(15)
        else:
            s += 'alpha_gas'.ljust(17) + ' = ' + '{}'.format(self.alpha_gas).rjust(15)

        s += '\n'
        
        if self.gap_loc is not None:
            s += 'gap_loc'.ljust(17) + ' = ' + '{}'.format(self.gap_loc/AU).rjust(15) + ' AU\n'
            s += 'gap_width'.ljust(17) + ' = ' + '{}'.format(self.gap_width/AU).rjust(15) + ' AU\n'
            s += 'gap_depth'.ljust(17) + ' = ' + '{}'.format(self.gap_depth).rjust(15)
            s += '\n'
            
        if self.M_iso is not None:
            s += 'M_seed'.ljust(17) + ' = ' + '{}'.format(self.M_seed/M_earth).rjust(15) + ' earth massess\n'
            s += 'M_iso'.ljust(17) + ' = ' + '{}'.format(self.M_iso/M_earth).rjust(15) + ' earth masses\n'

        # print vfrag
        
        if self.vfrag is None:
            s += 'vfrag'.ljust(17) + ' = ' + 'None'.rjust(15)
        elif hasattr(self.vfrag, '__call__'):
            s += 'vfrag'.ljust(17) + ' = ' + 'function'.rjust(15)
        else:
            s += 'vfrag'.ljust(17) + ' = ' + '{}'.format(self.vfrag).rjust(15) + ' cm/s'

        s += '\n' + 48 * '-' + '\n'
        return s

    def print_args(self):
        print(self.__str__())

    def write_args(self, fname=None):
        """
        Write out the simulation parameters to the file 'parameters.ini' in the
        folder specified in args.dir or to fname if that is not None
        """
        import configobj
        import os

        if fname is None:
            folder = self.dir
            fname = 'parameters.ini'
        else:
            fname = os.path.abspath(os.path.expanduser(fname))
            folder = os.path.dirname(fname)
            fname = os.path.basename(fname)

        if not os.path.isdir(folder):
            os.mkdir(folder)
        parser = configobj.ConfigObj()
        parser.filename = os.path.join(folder, fname)

        for name, _ in self.varlist:
            parser[name] = getattr(self, name)

        parser.write()

    def read_args(self):
        """
        Read in the simulation parameters from the file 'parameters.ini' in the
        folder specified in args.dir
        """
        import configobj
        import os
        import numpy as np

        parser = configobj.ConfigObj(self.dir + os.sep + 'parameters.ini')

        varlist = {v[0]: v[1] for v in self.varlist}

        for name, val in parser.items():
            if name not in varlist:
                print('Unknown Parameter:{}'.format(name))
                continue

            t = varlist[name]

            # process ints, bools, floats and lists of them

            if t in [int, bool, float]:
                if type(val) is list:
                    # lists
                    setattr(self, name, [t(v) for v in val])
                elif '[' in val:
                    # numpy arrays
                    val = val.replace('[', '').replace(']', '').replace('\n', ' ')
                    val = [t(v) for v in val.split()]
                    setattr(self, name, np.array(val))
                else:
                    try:
                        # plain values
                        setattr(self, name, t(val))
                    except:
                        try:
                            setattr(self, name, val)
                        except:
                            print('Could not convert variable \'{}\' with stored value {}'.format(name, t(val)))

            else:
                # stings and nones
                if val == 'None':
                    val = None
                else:
                    setattr(self, name, val)
