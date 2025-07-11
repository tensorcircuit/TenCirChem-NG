import numpy as np
import jax
from jax import numpy as jnp
from flax.training import train_state
import optax
from matplotlib import pyplot as plt
from tensorcircuit import Circuit

from tencirchem.molecule import h4, h2, h_chain
from tencirchem.applications.puccd_dnn.transformations import process_mol
from tencirchem.applications.puccd_dnn.expectation_sampling import get_prob, e_and_norm
from tencirchem.applications.puccd_dnn.models import get_circuit_nosym, MLP


# n_qubits = 4
# m = h4
n_qubits = 3
m = h_chain(3, charge=1)
# n_qubits = 2
# m = h2
active_space = None
aslst = None
fermion_circuit_state, hamiltonian, e_nuc, ci_mask, uccsd, puccd = process_mol(m, active_space, aslst)

nn_size = uccsd.n_qubits
size_factor = 2
# at least two layers
model = MLP([int(nn_size * size_factor)] * 2 + [1])


params = puccd.params
c1 = get_circuit_nosym(params, puccd)
c2 = Circuit(n_qubits)
for i in range(n_qubits):
    c2.ry(i, theta=0.2)


tx = optax.adamax(optax.linear_schedule(1e-2, 1e-3, 32000, 8000), b1=0.8, b2=0.99)

shots = 1 << 12
shots = None

e_min_list = []
e_list_list = []

for seed in range(10, 11):
    # for seed in [12]:
    e_list = []
    variables = model.init(jax.random.PRNGKey(seed), jnp.empty((1, nn_size)))
    # variables = params[seed]
    ts = train_state.TrainState.create(apply_fn=model.apply, params=variables, tx=tx)

    for stage in range(30):

        @jax.jit
        def train_step(ts):
            def loss_fn(params):
                def nn_func(batch):
                    return ts.apply_fn(params, batch).ravel()

                e, norm = e_and_norm(
                    c1,
                    c2,
                    n_qubits,
                    uccsd.n_elec // 2,
                    uccsd.h_qubit_op,
                    get_prob,
                    shots,
                    fun_nn=nn_func,
                    discard_eps=1e-3,
                )
                return e

            grad_fn = jax.value_and_grad(loss_fn)
            e, grads = grad_fn(ts.params)
            ts = ts.apply_gradients(grads=grads)
            return ts, e

        nsteps = 20
        # time1 = time.time()
        for i in range(nsteps):
            ts, e = train_step(ts)
            e_list.append(e)

        if stage % 3 == 0:
            print(e)
    plt.plot(np.array(e_list) - uccsd.e_fci)
    plt.show()
    e_min_list.append(np.min(e_list))
    e_list_list.append(e_list)
    print(seed, np.min(e_list))
print(e_min_list)
