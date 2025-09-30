#define PY_SSIZE_T_CLEAN
#include <Python.h>

typedef struct
{
    PyObject_HEAD
} PhyConstsFundObject;

static PyObject *
PhyConstsFund_get_caesium_frequency(PyObject *self, void *closure)
{
    (void)self;
    (void)closure;
    return PyLong_FromLongLong(9192631770LL);
}

static PyObject *
PhyConstsFund_get_vacuum_light_speed(PyObject *self, void *closure)
{
    (void)self;
    (void)closure;
    return PyLong_FromLongLong(299792458LL);
}

static PyObject *
PhyConstsFund_get_planck_constant(PyObject *self, void *closure)
{
    (void)self;
    (void)closure;
    return PyFloat_FromDouble(6.62607015e-34);
}

static PyObject *
PhyConstsFund_get_elementary_charge(PyObject *self, void *closure)
{
    (void)self;
    (void)closure;
    return PyFloat_FromDouble(1.602176634e-19);
}

static PyObject *
PhyConstsFund_get_boltzmann_constant(PyObject *self, void *closure)
{
    (void)self;
    (void)closure;
    return PyFloat_FromDouble(1.380649e-23);
}

static PyObject *
PhyConstsFund_get_avogadro_constant(PyObject *self, void *closure)
{
    (void)self;
    (void)closure;
    return PyFloat_FromDouble(6.02214076e23);
}

static PyObject *
PhyConstsFund_get_luminous_efficacy_kcd(PyObject *self, void *closure)
{
    (void)self;
    (void)closure;
    return PyLong_FromLong(683L);
}

static PyObject *
PhyConstsFund_get_gravitational_constant(PyObject *self, void *closure)
{
    (void)self;
    (void)closure;
    return PyFloat_FromDouble(6.6743e-11);
}

static PyObject *
PhyConstsFund_get_vacuum_electric_permittivity(PyObject *self, void *closure)
{
    (void)self;
    (void)closure;
    return PyFloat_FromDouble(8.8541878188e-12);
}

static PyObject *
PhyConstsFund_get_vacuum_magnetic_permeability(PyObject *self, void *closure)
{
    (void)self;
    (void)closure;
    return PyFloat_FromDouble(1.25663706127e-6);
}

static PyObject *
PhyConstsFund_get_atomic_mass_constant(PyObject *self, void *closure)
{
    (void)self;
    (void)closure;
    return PyFloat_FromDouble(1.66053906892e-27);
}

static PyObject *
PhyConstsFund_get_proton_mass(PyObject *self, void *closure)
{
    (void)self;
    (void)closure;
    return PyFloat_FromDouble(1.67262192595e-27);
}

static PyObject *
PhyConstsFund_get_neutron_mass(PyObject *self, void *closure)
{
    (void)self;
    (void)closure;
    return PyFloat_FromDouble(1.67492750056e-27);
}

static PyObject *
PhyConstsFund_get_electron_mass(PyObject *self, void *closure)
{
    (void)self;
    (void)closure;
    return PyFloat_FromDouble(9.1093837139e-31);
}

static PyGetSetDef PhyConstsFund_getset[] = {
    {"caesium_frequency", PhyConstsFund_get_caesium_frequency, NULL, "Caesium_frequency (SI) (read-only)", NULL},
    {"vacuum_light_speed", PhyConstsFund_get_vacuum_light_speed, NULL, "Speed of light in vacuum (SI) (read-only)", NULL},
    {"planck_constant", PhyConstsFund_get_planck_constant, NULL, "Planck constant (SI) (read-only)", NULL},
    {"elementary_charge", PhyConstsFund_get_elementary_charge, NULL, "Elementary charge (SI) (read-only)", NULL},
    {"boltzmann_constant", PhyConstsFund_get_boltzmann_constant, NULL, "Boltzmann constant (SI) (read-only)", NULL},
    {"avogadro_constant", PhyConstsFund_get_avogadro_constant, NULL, "Avogadro constant (SI) (read-only)", NULL},
    {"luminous_efficacy_kcd", PhyConstsFund_get_luminous_efficacy_kcd, NULL, "luminous efficacy Kcd (SI) (read-only)", NULL},
    {"gravitational_constant", PhyConstsFund_get_gravitational_constant, NULL, "Gravitational constant (SI) (read-only)", NULL},
    {"vacuum_electric_permittivity", PhyConstsFund_get_vacuum_electric_permittivity, NULL, "Vacuum electric permittivity (SI) (read-only)", NULL},
    {"vacuum_magnetic_permeability", PhyConstsFund_get_vacuum_magnetic_permeability, NULL, "Vacuum magnetic permeability (SI) (read-only)", NULL},
    {"atomic_mass_constant", PhyConstsFund_get_atomic_mass_constant, NULL, "Atomic mass constant (SI) (read-only)", NULL},
    {"proton_mass", PhyConstsFund_get_proton_mass, NULL, "Proton mass (SI) (read-only)", NULL},
    {"neutron_mass", PhyConstsFund_get_neutron_mass, NULL, "Neutron mass (SI) (read-only)", NULL},
    {"electron_mass", PhyConstsFund_get_electron_mass, NULL, "Electron mass (SI) (read-only)", NULL},
    {NULL}};

static PyObject *singleton_instance = NULL;

static PyObject *
PhyConstsFund_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    (void)args;
    (void)kwds;

    if (singleton_instance)
    {
        Py_INCREF(singleton_instance);
        return singleton_instance;
    }

    singleton_instance = type->tp_alloc(type, 0);
    return singleton_instance;
}

static PyTypeObject PhyConstsFundType = {
    PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "parameters.phy_consts_fund.PhyConsts",
    .tp_basicsize = sizeof(PhyConstsFundObject),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Read-only container for fundamental physical constants",
    .tp_getset = PhyConstsFund_getset,
    .tp_new = PhyConstsFund_new,
};

static PyModuleDef phyconstsfund_module = {
    PyModuleDef_HEAD_INIT,
    "parameters.phy_consts_fund",
    "C extension exposing fundamental physical constants as `consts_fund`.",
    -1,
    NULL, NULL, NULL, NULL, NULL};

PyMODINIT_FUNC
PyInit_phy_consts_fund(void)
{
    PyObject *m;

    if (PyType_Ready(&PhyConstsFundType) < 0)
        return NULL;

    m = PyModule_Create(&phyconstsfund_module);
    if (m == NULL)
        return NULL;

    singleton_instance = PyObject_CallObject((PyObject *)&PhyConstsFundType, NULL);
    if (singleton_instance == NULL)
    {
        Py_DECREF(m);
        return NULL;
    }

    if (PyModule_AddObject(m, "consts_fund", singleton_instance) < 0)
    {
        Py_DECREF(singleton_instance);
        Py_DECREF(m);
        return NULL;
    }

    PyObject *all = PyList_New(0);
    if (all)
    {
        PyObject *name = PyUnicode_FromString("consts_fund");
        if (name)
        {
            if (PyList_Append(all, name) == 0)
            {
                if (PyModule_AddObject(m, "__all__", all) < 0)
                {
                    Py_DECREF(all);
                }
            }
            else
            {
                Py_DECREF(all);
            }
            Py_DECREF(name);
        }
        else
        {
            Py_DECREF(all);
        }
    }

    return m;
}