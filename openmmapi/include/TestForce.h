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
     *          U = p1 * p2 / r^2
     */
    TestForce();
    /**
     * Add particle
     * @param factor     pre-factor
     */
    void addParticle(double factor);
    /**
     * Get num of particles
     * @return           num
     */
    int getNumParticles();
    /**
     * Set particle parameter
     * @param index      index
     * @param factor     factor
     */
    void setParticleParameter(int index, double factor);
    /**
     * Get particle parameter
     * @param index      index
     * @return           factor
     */
    double getParticleParameter(int index);
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
    bool usesPeriodicBoundaryConditions() const;
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
    std::vector<double> params;
};

} // namespace TestPlugin

#endif /*OPENMM_TESTFORCE_H_*/