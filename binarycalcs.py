import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.constants import G, M_sun, au
from astropy.units.core import UnitConversionError

def keplerian_binary(givenquant):
    '''Return equivalency for Keplerian binary orbit.

    Parameters
    ----------
    givenquant : `~astropy.units.Quantity`
        `astropy.units.Quantity` associated with the parameter of the orbit
        that is fixed for this conversion (e.g. to convert between period and
        semimajor axis, this should be a mass quanitity).
    '''
    # Finding a pythonic way to to cycle through the three potential choices
    # for givenquant has been difficult. This seems to follow the rule of EAFP
    # best. First I will assume that givenquant is a mass, then a semimajor
    # axis, then a period.
    try:
        fixedmass = givenquant.to(u.solMass)
    except UnitConversionError:
        try:
            fixedsemimajor = givenquant.to(u.AU)
        except UnitConversionError:
            try:
                fixedperiod = givenquant.to(u.year).value
            except UnitConversionError:
                # If it's neither a mass, length, or year, then the wrong
                # quantity was given.
                raise ValueError(
                    "The fixed quantity must be either a mass, time interval, "
                    "or length.")
            else:
                # givenquant is a time
                fromunit = u.solMass
                tounit = u.AU
                def fromfunction(M):
                    return (M * fixedperiod**2)**(1/3)
                def tofunction(a):
                    return a**3 / fixedperiod**2
        else: 
            # givenquant is a length
            fromunit = u.solMass
            tounit = u.year
            def fromfunction(M):
                return (fixedsemimajor**3 / M)**(1/2)
            def tofunction(P):
                return fixedsemimajor**3 / P**2
    else:
        # givenquant is a mass
        fromunit = u.year
        tounit = u.AU
        def fromfunction(P):
            return (P**2 * fixedmass)**(1/3)
        def tofunction(a):
            return (a**3 / fixedmass)**(1/2)

    equiv = [
        (fromunit, tounit, fromfunction, tofunction)]
    return equiv
            

def calc_velocity_of_binary(masses, period, mass_ratio):
    '''Returns the orbital velocity of a binary specified by mass and period.

    The masses should be the total mass of the system and the period should be
    the orbital period of the system.
    '''
    vel = ((2 * np.pi * G * masses / period)**(1/3) * mass_ratio / 
           (1 + mass_ratio))
    try:
        return vel.to(u.km/u.s)
    except u.UnitConversionError as e:
        raise TypeError("Arguments should be Astropy Quantities with "
                        "appropriate units") 
