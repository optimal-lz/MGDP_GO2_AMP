"""Fixed-size replay for policy motion transitions."""

import torch


class AMPReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, device: str,
                 seed: int = 0):
        if capacity <= 0 or state_dim <= 0:
            raise ValueError("positive capacity and state_dim required")
        self.capacity = capacity
        self.states = torch.empty(capacity, state_dim, device=device)
        self.next_states = torch.empty_like(self.states)
        self.cursor = 0
        self.size = 0
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(int(seed))

    def insert(self, state, next_state):
        if state.shape != next_state.shape or state.ndim != 2 or state.shape[1] != self.states.shape[1]:
            raise ValueError("invalid AMP transition dimensions")
        count = state.shape[0]
        if count >= self.capacity:
            state, next_state = state[-self.capacity:], next_state[-self.capacity:]
            count = self.capacity
        positions = (torch.arange(count, device=self.states.device) + self.cursor) % self.capacity
        self.states[positions] = state.detach()
        self.next_states[positions] = next_state.detach()
        self.cursor = (self.cursor + count) % self.capacity
        self.size = min(self.capacity, self.size + count)

    def sample(self, batch_size):
        if self.size == 0:
            raise ValueError("cannot sample empty AMP replay")
        indices = torch.randint(
            self.size, (batch_size,), device=self.states.device,
            generator=self.generator)
        return self.states[indices], self.next_states[indices]

    def state_dict(self):
        return {
            "capacity": self.capacity,
            "state_dim": self.states.shape[1],
            "states": self.states.clone(),
            "next_states": self.next_states.clone(),
            "cursor": self.cursor,
            "size": self.size,
            "generator_state": self.generator.get_state().clone(),
        }

    def validate_state_dict(self, state):
        required = {
            "capacity", "state_dim", "states", "next_states", "cursor", "size",
            "generator_state",
        }
        missing = required.difference(state)
        if missing:
            raise ValueError(f"AMP replay state missing fields: {sorted(missing)}")

        capacity = int(state["capacity"])
        state_dim = int(state["state_dim"])
        expected_shape = (self.capacity, self.states.shape[1])
        if capacity != self.capacity:
            raise ValueError(
                f"AMP replay capacity mismatch: checkpoint {capacity}, current {self.capacity}"
            )
        if state_dim != self.states.shape[1]:
            raise ValueError(
                f"AMP replay state_dim mismatch: checkpoint {state_dim}, "
                f"current {self.states.shape[1]}"
            )
        if tuple(state["states"].shape) != expected_shape:
            raise ValueError("AMP replay states shape mismatch")
        if tuple(state["next_states"].shape) != expected_shape:
            raise ValueError("AMP replay next_states shape mismatch")

        cursor = int(state["cursor"])
        size = int(state["size"])
        if not 0 <= cursor < self.capacity:
            raise ValueError("AMP replay cursor out of range")
        if not 0 <= size <= self.capacity:
            raise ValueError("AMP replay size out of range")
        if not torch.is_tensor(state["generator_state"]):
            raise ValueError("AMP replay generator state is invalid")
        return cursor, size

    def load_state_dict(self, state):
        cursor, size = self.validate_state_dict(state)
        states = state["states"].to(device=self.states.device, dtype=self.states.dtype)
        next_states = state["next_states"].to(
            device=self.next_states.device, dtype=self.next_states.dtype
        )
        self.states.copy_(states)
        self.next_states.copy_(next_states)
        self.cursor = cursor
        self.size = size
        self.generator.set_state(state["generator_state"].cpu())
