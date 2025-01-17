  
%module openmmnltest

%import(module="simtk.openmm") "swig/OpenMMSwigHeaders.i"
%include "swig/typemaps.i"
%include <std_string.i>
%include <std_vector.i>

%{
#include "TestForce.h"
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
#include "openmm/RPMDIntegrator.h"
#include "openmm/RPMDMonteCarloBarostat.h"
%}

%pythoncode %{
import simtk.openmm as mm
import simtk.unit as unit
%}

/*
 * Convert C++ exceptions to Python exceptions.
*/
%exception {
    try {
        $action
    } catch (std::exception &e) {
        PyErr_SetString(PyExc_Exception, const_cast<char*>(e.what()));
        return NULL;
    }
}

%feature("shadow") TestPlugin::TestForce::TestForce %{
    def __init__(self, *args):
        this = _openmmnltest.new_TestForce()
        try:
            self.this.append(this)
        except Exception:
            self.this = this
%}

namespace std {
  %template(IntVector) vector<int>;
}

namespace TestPlugin {

class TestForce : public OpenMM::Force {
public:
    TestForce();
    void addParticle(double factor);
    int getNumParticles() const;
    void setParticleParameter(int index, double factor);
    double getParticleParameter(int index) const;
    double getCutoffDistance() const;
    void setCutoffDistance(double cutoff);
    bool usesPeriodicBoundaryConditions() const;
    void setUsesPeriodicBoundaryConditions(bool ifPeriod);
    void addExclusion(int particle1, int particle2);
    int getNumExclusions() const;
    /*
     * Add methods for casting a Force to a TestForce.
    */
    %extend {
        static TestPlugin::TestForce& cast(OpenMM::Force& force) {
            return dynamic_cast<TestPlugin::TestForce&>(force);
        }

        static bool isinstance(OpenMM::Force& force) {
            return (dynamic_cast<TestPlugin::TestForce*>(&force) != NULL);
        }
    }
};

}