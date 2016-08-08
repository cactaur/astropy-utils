import astropy.units as u


def list_sorted(lst, ascending=True):
    '''Checks to see if the given list is sorted.

    Function returns true or false depending on if the list is sorted. It by
    default checks to see if it is sorted ascending. If the ascending keyword
    is false, then it will check to see if it is sorted descending. 
    '''
    # Empty list is automatically sorted.
    if lst is []:
        return True

    if ascending:
        return all(lst[i] <= lst[i+1] for i in range(len(lst)-1))
    else:
        return all(lst[i] >= lst[i+1] for i in range(len(lst)-1))


class BrokenPowerLawIMF(object):
    '''Class representing a generic multiply-broken power law.'''

    def __init__(self, indices, breakmasses, highM=80*u.solMass,
        lowM=0.08*u.solMass):
        '''Create a broken power law.

        The indices should be a list of indices (non-negative). Breakmasses
        should be a list of masses with one less element than indices.
        Breakmasses should also be a sorted list so that indices make sense.
        '''
        if list_sorted(breakmasses, False):
            pass
        elif list_sorted(breakmasses, True):
            # In the spirit of IMFs, breakmasses will start at high masses, and
            # descend to low masses.
            breakmasses.reverse()
            indices.reverse()
        else:
            raise ValueError("Breakmasses are not sorted")

        self.highM = highM
        self.lowM = lowM
        self.break_masses = breakmasses
        self.indices = indices

    def _integrate(self, highbound, lowbound):
        '''Integrates the unnormalized IMF.

        This function is essentially supposed to help perform the piecewise
        integration of the broken power law.
        '''
        # Running sum of the integral for each piecewise region.
        piece_sum = 0 
        # Masses which contain boundaries and break points.
        special_masses = self.break_masses 
        for i,ind in enumerate(self.indices):
            # This can probably be made into a more streamlined algorithm.
            if i == 0:
                uplim = highbound
                # If there are no special masses, then just set lowlim to be
                # lowbound.
                try:
                    lowlim = max(lowbound, special_masses[i])
                    print("Massbreaks: {0}, {1}".format(highbound,
                                                        special_masses[i]))
                except IndexError:
                    lowlim = lowbound
                    print("Massbreaks: {0}, {1}".format(highbound,
                                                        lowbound))
            elif i == len(self.indices)-1:
                uplim = min(highbound, special_masses[i-1])
                lowlim = lowbound
                print("Massbreaks: {0}, {1}".format( special_masses[-1],
                                                    lowbound))
            else:
                uplim = min((highbound, special_masses[i-1]))
                lowlim = max((lowbound, special_masses[i]))

                print("Massbreaks: {0}, {1}".format(special_masses[i-1],
                                                    special_masses[i]))
            print("Upper limit: {0}".format(uplim))
            print("Lower limit: {0}".format(lowlim))

            if uplim > lowlim:
                print("Counts!")
                piece_sum += (uplim.value**(1-ind) - lowlim.value**(1-ind)) / (1-ind)

        return piece_sum

    def number_fraction(self, highbound=None, lowbound=None, highM=None,
                        lowM=None):
        '''
        Calculates the number fraction of stars between highbound and lowbound.

        Either one of highbound or lowbound has to be given. If one is missing,
        then it will be substituted for either highM or lowM. If both are
        missing, this function will raise an error rather than return 1.
        '''
        if highbound is None and lowbound is None:
            raise ValueError("Need at least one bound for IMF")
        elif highbound is None and lowbound is not None:
            highbound = self.highM
        elif highbound is not None and lowbound is None:
            lowbound = self.lowM

        if lowM is None:
            lowM = self.lowM
        if highM is None:
            highM = self.highM

        if highbound > highM or lowbound < lowM:
            raise ValueError("Integration limits must be within allowable "
                             "mass range.")
                             

        frac = (
            self._integrate(highbound, lowbound) / self._integrate(highM, lowM)) 
        return frac

class PowerLawIMF(BrokenPowerLawIMF):
    '''Class representing an IMF with a generic power-law.'''

    def __init__(self, index, highM=80*u.solMass, lowM=0.08*u.solMass):
        '''
        Create a power law with a given index.
        '''
        super().__init__([2.35], [], lowM=lowM, highM=highM)

class SalpeterIMF(PowerLawIMF):
    """
    Class for doing operations with a Salpeter IMF.
    """
    def __init__(self, lowM=0.08*u.solMass, highM=80*u.solMass):
        super().__init__(2.35, lowM=lowM, highM=highM)

class KroupaIMF(BrokenPowerLawIMF):
    '''Class representing a Kroupa IMF'''
    def __init__(self, highM=80*u.solMass, lowM=0.08*u.solMass):
        '''Create a Kroupa IMF calculator'''
        super().__init__([2.3, 1.3, 0.3], [0.5*u.solMass, 0.08*u.solMass], 
                         highM, lowM)

