# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: nonecheck=False
# cython: profile=False

from cpython.list cimport PyList_GET_ITEM
cimport numpy as cnp
from libc.stdlib cimport rand
from libc.math cimport sqrt, sin

def step_all(list envs):
    cdef int n = len(envs)
    for i in range(n):
        (<CHighway>PyList_GET_ITEM(envs, i)).step()

cdef class CHighway:
    cdef:
        float[:,:] observations
        float[:,:] actions
        float[:] rewards
        float[:] veh_positions
        float[:] veh_speeds
        float[:] veh_accels
        float[:] veh_gaps
        float[:] veh_lead_speeds
        float car_length
        float idm_v0
        float idm_T
        float idm_a
        float idm_b
        float idm_delta
        float idm_s0
        float t
        float dt
        int n_vehicles

    def __init__(self, 
                 cnp.ndarray observations,
                 cnp.ndarray actions,
                 cnp.ndarray rewards,
                 cnp.ndarray veh_positions,
                 cnp.ndarray veh_speeds,
                 cnp.ndarray veh_accels,
                 cnp.ndarray veh_gaps,
                 cnp.ndarray veh_lead_speeds,
                 float car_length,
            ):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.veh_positions = veh_positions
        self.veh_speeds = veh_speeds
        self.veh_accels = veh_accels
        self.veh_gaps = veh_gaps
        self.veh_lead_speeds = veh_lead_speeds
        self.car_length = car_length

        self.idm_v0 = 45
        self.idm_T = 1.0
        self.idm_a = 1.8
        self.idm_b = 2.0
        self.idm_delta = 4
        self.idm_s0 = 1.0

        self.t = 0
        self.dt = 0.05

        self.n_vehicles = len(self.veh_positions)

    cdef void compute_observations(self):
        self.observations[0][0] = self.veh_speeds[1] / 30.0
        self.observations[0][1] = self.veh_lead_speeds[1] / 30.0
        self.observations[0][2] = self.veh_gaps[1] / 200.0

    cpdef void reset(self):
        cdef int i
        for i in range(self.n_vehicles):
            self.veh_positions[i] = - i * 30
            self.veh_speeds[i] = 20
        self.t = 0

        self.compute_observations()

    cdef float idm_accel(self, float this_vel, float lead_vel, float headway):
        if lead_vel is None:
            s_star = 0
        else:
            s_star = self.idm_s0 + max(0, this_vel * self.idm_T + this_vel * (this_vel - lead_vel) / (2 * sqrt(self.idm_a * self.idm_b)))
        accel = self.idm_a * (1 - (this_vel / self.idm_v0) ** self.idm_delta - (s_star / headway) ** 2)
        return accel

    cdef void step(self):
        cdef:
            int i
            float accel

        for i in range(self.n_vehicles):
            if i > 0:
                self.veh_lead_speeds[i] = self.veh_speeds[i - 1]
                self.veh_gaps[i] = self.veh_positions[i - 1] - self.veh_positions[i] - self.car_length
            if i == 1:
                # AV idx (TODO harcoded)
                self.veh_accels[i] = min(max(self.actions[0][0], -3), 1.5)
                self.veh_speeds[i] += self.veh_accels[i] * self.dt
            elif i > 0:
                self.veh_accels[i] = self.idm_accel(self.veh_speeds[i], self.veh_lead_speeds[i], self.veh_gaps[i])
                self.veh_speeds[i] += self.veh_accels[i] * self.dt
            else:
                self.veh_speeds[i] = 20 + min(max(30 * sin(self.t / 8), -10), 10)
            self.veh_speeds[i] = min(max(self.veh_speeds[i], 0), 30)

        for i in range(self.n_vehicles):
            self.veh_positions[i] += self.veh_speeds[i] * self.dt

        self.t += self.dt
        
        # TODO hardcoded for a single agent
        self.rewards[0] = 0
        for i in range(1, self.n_vehicles):
            self.rewards[0] -= 0.1 * self.veh_accels[i] * self.veh_accels[i]
        if self.veh_gaps[1] > 200 or self.veh_gaps[1] < 1:
            self.rewards[0] -= 10

        if self.t >= 100:
            self.reset()  # TODO SET DONE

        self.compute_observations()
