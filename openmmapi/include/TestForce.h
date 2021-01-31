#ifndef OPENMM_TESTFORCE_H_
#define OPENMM_TESTFORCE_H_

#include "openmm/Context.h"
#include "openmm/Force.h"
#include <string>
#include <utility>
#include <vector>
#include <set>

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
    int getNumParticles() const;
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
    double getParticleParameter(int index) const;
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
    /**
     * Add exclusion pair.
     */
    void addExclusion(int particle1, int particle2);
    /**
     * Get exclusion particles. (Not avaliable in Python wrapper)
     */
    void getExclusionParticles(int index, int& paritcle1, int& particle2);
    /**
     * Get number of exclusions.
     */
    int getNumExclusions() const; 

protected:
    OpenMM::ForceImpl* createImpl() const;
private:
    double cutoffDistance;
    bool ifPBC;
    std::vector<double> params;
    std::vector<std::pair<int,int>> exclusions;
};

} // namespace TestPlugin

#endif /*OPENMM_TESTFORCE_H_*/