#ifndef OPENMM_TESTFORCE_H_
#define OPENMM_TESTFORCE_H_

#include "openmm/Context.h"
#include "openmm/Force.h"
#include <string>
#include <utility>
#include <vector>

namespace TestPlugin {

/**
 * This class implements a simple force field to test the usage of neighborlist. 
 */

class TestForce : public OpenMM::Force {
public:
    /**
     * Create a TestForce like:
     *          U = 100 / r^2
     */
    TestForce();
    /**
     * Get Cutoff Distance
     * @return            cutoff
     */
    double getCutoffDistance() const;
    /**
     * Set the cutoff distance
     * @param cutoff      cutoff
     */
    void setCutoffDistance(double cutoff);
    /**
     * Return if PBC is used in this force. Default is no.
     * @return             whether PBC system
     */
    bool usesPeriodicBoundaryConditions();
    /**
     * Set if using PBC in the system.
     * @param ifPeriod     if use PBC
     */
    void setUsesPeriodicBoundaryConditions(bool ifPeriod);

protected:
    OpenMM::ForceImpl* createImpl() const;
private:
    double cutoffDistance;
    bool ifPBC;
};

} // namespace TestPlugin

#endif /*OPENMM_TESTFORCE_H_*/