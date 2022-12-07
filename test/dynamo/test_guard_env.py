# Owner(s): ["module: dynamo"]
import torch

import torch._dynamo.test_case
import torch._dynamo.testing

from torch.fx.experimental.guard_env import DuplicateInputs, GUARD_ENV, GuardEnvExpr

# Note please keep these tests unit tests - for end to end tests around compilation,
# see test_arg_dupe_via_dynamo_recompiles in test_aot_autograd.
class GuardEnvTests(torch._dynamo.test_case.TestCase):
    def test_guard_env_register_duplicates(self):
        tensor_x = torch.tensor([0.5, 0.5, 0.5])
        tensor_y = tensor_x

        GUARD_ENV.associate(tensor_x, 'tensor_x')
        GUARD_ENV.associate(tensor_y, 'tensor_y')

        GUARD_ENV.register_duplicates(tensor_x, tensor_y)

        guards = GUARD_ENV.get_guards()

        self.assertEqual(guards[0], DuplicateInputs(arg_a='tensor_x', arg_b='tensor_y'))

    def test_guard_env_throws_without_associate(self):
        tensor_x = torch.tensor([0.5, 0.5, 0.5])
        tensor_y = tensor_x

        with self.assertRaisesRegex(AssertionError, 
        """Tensor not found - did you forget to call .associate()"""):
            GUARD_ENV.register_duplicates(tensor_x, tensor_y), 

    def test_guard_env_throws_dup_but_not_dup(self):
        tensor_x = torch.tensor([0.5, 0.5, 0.5])
        tensor_y = torch.tensor([0.5, 0.5, 0.5])

        GUARD_ENV.associate(tensor_x, 'tensor_x')
        GUARD_ENV.associate(tensor_y, 'tensor_y')

        with self.assertRaisesRegex(AssertionError, """Register_duplicates args must pass identity check."""):
            GUARD_ENV.register_duplicates(tensor_x, tensor_y)

