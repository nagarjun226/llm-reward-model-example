"""
Object Sorting Task: A Clean Demonstration of RL, LLM, and LLM+RL
==================================================================

This example demonstrates the key differences between:
1. Pure RL - Learns weight threshold through trial and error
2. Static LLM - Uses visual reasoning but ignores rewards
3. LLM + Reward Model - Combines reasoning with learning from feedback

Task: Sort objects by weight into buckets A, B, or C
- Bucket A: Light objects (10-inch pot)
- Bucket B: Heavy objects (30-inch pot)  
- Bucket C: "I don't know" (safe option)

Rewards:
- +1 for correct placement in A or B
- -1 for incorrect placement in A or B
- 0 for placement in C

The weight threshold is unknown (10kg in our case).
"""

# Standard scientific computing and data manipulation libraries
import numpy as np              # For numerical operations, random number generation, and array handling
import random                   # For basic randomization functions (exploring actions, generating objects)
import matplotlib.pyplot as plt # For creating visualizations and plots of agent performance
from collections import defaultdict, deque  # defaultdict for auto-initializing dicts, deque for efficient queues
from typing import Dict, Tuple, List       # Type hints for better code documentation and IDE support
from dataclasses import dataclass          # For creating simple data classes with automatic methods
import json                     # For parsing JSON responses from LLM API calls
from openai import OpenAI       # OpenAI API client for interacting with GPT models
import os                       # For accessing environment variables (like API keys)


@dataclass
class Object:
    """
    A data class representing an object to be sorted.
    
    This is a simple container that holds all the information about each object
    that gets presented to the sorting agents. Using @dataclass automatically
    generates __init__, __repr__, and other special methods.
    """
    name: str          # Human-readable name (e.g., "laptop", "feather", "bowling ball")
    weight: float      # The actual weight in kilograms - this is the ground truth the agents must learn
    description: str   # Natural language description for the LLM agents to reason about
    
    def __repr__(self):
        """Custom string representation showing name and weight for debugging purposes"""
        return f"{self.name} ({self.weight}kg)"


class SortingEnvironment:
    """
    The environment that generates objects and provides rewards.
    
    This acts as the "world" that the agents interact with. It:
    1. Generates random objects with varying weights
    2. Calculates rewards based on correct/incorrect bucket placement
    3. Hides the true weight threshold from agents (they must learn it)
    """
    
    def __init__(self, weight_threshold: float = 10.0):
        """
        Initialize the environment with a hidden weight threshold.
        
        Args:
            weight_threshold: The secret boundary between light (A) and heavy (B) objects.
                            Objects below this go in bucket A, above go in bucket B.
                            Agents don't know this value and must learn it!
        """
        self.weight_threshold = weight_threshold  # This is the secret the agents must discover
        
        # Descriptions of the three buckets available to agents
        # These descriptions help LLM agents understand the physical constraints
        self.bucket_descriptions = {
            'A': "Small 10-inch pot (for lighter objects)",    # Should hold light objects
            'B': "Large 30-inch pot (for heavier objects)",   # Should hold heavy objects  
            'C': "Middle bucket labeled 'I don't know'"       # Safe choice when uncertain
        }
        
    def generate_object(self) -> Object:
        """
        Generate a random object to sort with realistic weight distribution.
        
        This method creates diverse objects with weights that cluster around the
        threshold (10kg) to make the learning problem challenging but realistic.
        """
        # Create diverse weights using exponential + normal distribution
        # This creates a realistic distribution with many objects near the threshold
        weight = np.random.exponential(8.0) + np.random.normal(2.0, 0.5)
        weight = max(0.1, weight)  # Ensure positive weight (no negative-weight objects!)
        
        # Create realistic object names and descriptions based on weight ranges
        # This gives LLM agents intuitive clues about object weights
        if weight < 5:  # Very light objects
            names = ["feather", "pencil", "paper", "plastic cup", "toy car"]
            desc_template = "A small, lightweight {} that looks easy to lift"
        elif weight < 10:  # Medium weight objects (near the threshold!)
            names = ["book", "laptop", "water bottle", "shoe", "small bag"]
            desc_template = "A medium-sized {} of moderate weight"
        elif weight < 20:  # Heavy objects
            names = ["bowling ball", "dumbbell", "toolbox", "backpack full of books", "small television"]
            desc_template = "A heavy {} that requires effort to lift"
        else:  # Very heavy objects
            names = ["car battery", "concrete block", "large rock", "metal chair", "full suitcase"]
            desc_template = "A very heavy {} that's difficult to move"
        
        # Randomly select a name from the appropriate category
        name = random.choice(names)
        # Create the description by filling in the template
        description = desc_template.format(name)
        
        # Return a complete Object with all the information
        return Object(name=name, weight=weight, description=description)
    
    def get_reward(self, obj: Object, bucket: str) -> float:
        """
        Calculate reward for placing object in bucket.
        
        This is the key feedback mechanism that agents use to learn.
        The reward structure encourages correct classification while
        allowing agents to play it safe when uncertain.
        
        Args:
            obj: The object that was sorted
            bucket: The bucket choice ('A', 'B', or 'C')
            
        Returns:
            +1.0 for correct placement in A or B
            -1.0 for incorrect placement in A or B  
             0.0 for using the "I don't know" bucket C
        """
        if bucket == 'C':
            return 0.0  # Safe choice - no penalty for admitting uncertainty
        
        # Determine the correct bucket based on the hidden threshold
        correct_bucket = 'A' if obj.weight < self.weight_threshold else 'B'
        
        # Give positive reward for correct classification, negative for wrong
        if bucket == correct_bucket:
            return 1.0   # Correct! Agent learned something useful
        else:
            return -1.0  # Wrong! This should discourage similar future choices


class PureRLAgent:
    """
    Pure RL agent that only sees weight and learns through rewards.
    
    This agent implements Q-learning with linear function approximation.
    It starts completely blind (no prior knowledge) and learns the weight
    threshold purely through trial and error using reward feedback.
    
    Key characteristics:
    - Only sees numerical weight (no descriptions or reasoning)
    - Uses epsilon-greedy exploration strategy
    - Learns Q-values using temporal difference learning
    - Represents Q(s,a) as linear combination of features
    """
    
    def __init__(self, learning_rate: float = 0.1):
        """
        Initialize the RL agent with Q-learning parameters.
        
        Args:
            learning_rate: How fast to update Q-values (alpha in Q-learning)
        """
        self.learning_rate = learning_rate  # Controls how much we update weights each step
        
        # Q-learning with linear function approximation
        # Instead of a Q-table, we use features [weight, weight^2, bias] for each action
        # Q(s,a) = w_a^T * features(s), where w_a are the weights for action a
        self.weights = {
            'A': np.random.randn(3) * 0.01,  # Small random weights for bucket A
            'B': np.random.randn(3) * 0.01,  # Small random weights for bucket B
            'C': np.random.randn(3) * 0.01   # Small random weights for bucket C
        }
        
        # Epsilon-greedy exploration parameters
        self.epsilon = 1.0          # Start with 100% exploration (random actions)
        self.epsilon_decay = 0.995  # Gradually reduce exploration over time
        self.epsilon_min = 0.01     # Always maintain 1% exploration
        
        # Memory to store recent experiences for analysis
        self.memory = deque(maxlen=1000)  # Keep last 1000 (state, action, reward) tuples
        
        # Track learned threshold over time (for plotting)
        self.threshold_history = []
        
    def get_features(self, weight: float) -> np.ndarray:
        """
        Extract features from weight for function approximation.
        
        We use polynomial features [weight, weight^2, 1] to allow the
        Q-function to learn non-linear decision boundaries. The weight
        is normalized by 20 to keep feature values in a reasonable range.
        
        Args:
            weight: Object weight in kg
            
        Returns:
            Feature vector [normalized_weight, normalized_weight^2, bias_term]
        """
        normalized_weight = weight / 20.0  # Normalize to [0, ~1] range
        return np.array([
            normalized_weight,           # Linear term
            normalized_weight ** 2,      # Quadratic term (for non-linear boundaries)
            1.0                         # Bias term (constant)
        ])
    
    def get_q_values(self, weight: float) -> Dict[str, float]:
        """
        Calculate Q-values for each action given object weight.
        
        Q(s,a) = weights[a] · features(s)
        This gives us the expected future reward for taking action a in state s.
        
        Args:
            weight: Object weight in kg
            
        Returns:
            Dictionary mapping each action to its Q-value
        """
        features = self.get_features(weight)  # Get feature representation
        return {
            action: np.dot(self.weights[action], features)  # Linear combination
            for action in ['A', 'B', 'C']
        }
    
    def act(self, obj: Object, training: bool = True) -> str:
        """
        Choose which bucket to use based on epsilon-greedy policy.
        
        During training, the agent balances exploration (trying random actions
        to discover new information) with exploitation (using current knowledge
        to maximize reward).
        
        Args:
            obj: Object to sort (only weight is used)
            training: Whether we're in training mode (affects exploration)
            
        Returns:
            Bucket choice: 'A', 'B', or 'C'
        """
        # RL agent only sees numerical weight (no descriptions or reasoning)
        weight = obj.weight
        
        if training and random.random() < self.epsilon:
            # EXPLORE: Try a random action to gather new information
            # This is crucial early in training when we know nothing
            return random.choice(['A', 'B', 'C'])
        else:
            # EXPLOIT: Use current knowledge to choose best action
            q_values = self.get_q_values(weight)  # Get expected rewards
            # Choose action with highest Q-value (greedy policy)
            return max(q_values.items(), key=lambda x: x[1])[0]
    
    def get_learned_threshold(self) -> float:
        """
        Calculate the learned threshold where Q(weight, 'A') = Q(weight, 'B').
        
        The threshold is the weight where the agent is indifferent between buckets A and B.
        We solve: w_A^T * features(threshold) = w_B^T * features(threshold)
        
        Returns:
            Estimated threshold weight in kg
        """
        # Features are [weight/20, (weight/20)^2, 1]
        # We want to find weight where Q_A(weight) = Q_B(weight)
        # w_A[0] * (weight/20) + w_A[1] * (weight/20)^2 + w_A[2] = w_B[0] * (weight/20) + w_B[1] * (weight/20)^2 + w_B[2]
        
        # Let u = weight/20, then solve: (w_A[1] - w_B[1]) * u^2 + (w_A[0] - w_B[0]) * u + (w_A[2] - w_B[2]) = 0
        diff_weights = self.weights['A'] - self.weights['B']
        a, b, c = diff_weights[1], diff_weights[0], diff_weights[2]  # Coefficients for u^2, u, constant
        
        # Solve quadratic equation: au^2 + bu + c = 0
        if abs(a) < 1e-10:  # Linear case: bu + c = 0
            if abs(b) < 1e-10:
                return 10.0  # Default if no clear preference
            u = -c / b
        else:  # Quadratic case
            discriminant = b**2 - 4*a*c
            if discriminant < 0:
                return 10.0  # Default if no real solution
            sqrt_disc = np.sqrt(discriminant)
            u1, u2 = (-b + sqrt_disc) / (2*a), (-b - sqrt_disc) / (2*a)
            
            # Convert back to weight: weight = 20 * u
            t1, t2 = 20 * u1, 20 * u2
            
            # Choose the threshold in reasonable range [0, 30]
            candidates = [t for t in [t1, t2] if 0 <= t <= 30]
            u = candidates[0] / 20.0 if candidates else 0.5  # Default to 10kg if no valid solution
            
        # Convert u back to weight
        threshold = 20.0 * u
        return np.clip(threshold, 0.1, 30.0)  # Keep in reasonable range
    
    def update(self, obj: Object, action: str, reward: float):
        """
        Update Q-values based on received reward using temporal difference learning.
        
        This implements the core Q-learning update rule:
        Q(s,a) ← Q(s,a) + α * [reward - Q(s,a)]
        
        Since we use function approximation, we update the weights that generate Q-values.
        
        Args:
            obj: The object that was sorted
            action: The action that was taken ('A', 'B', or 'C')
            reward: The reward received for this action
        """
        features = self.get_features(obj.weight)  # Get features for this state
        
        # Calculate temporal difference error
        current_q = np.dot(self.weights[action], features)  # Current Q-value estimate
        td_error = reward - current_q  # How far off was our prediction?
        
        # Update weights using gradient descent
        # Δw = α * td_error * features (this moves Q-value toward observed reward)
        self.weights[action] += self.learning_rate * td_error * features
        
        # Store experience for potential analysis
        self.memory.append((obj.weight, action, reward))
        
        # Track learned threshold for plotting
        self.threshold_history.append(self.get_learned_threshold())
        
        # Decay exploration rate over time (explore less as we learn more)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class StaticLLMAgent:
    """
    LLM agent that uses reasoning but ignores rewards.
    
    This agent demonstrates pure "System 2" thinking - it uses language
    and reasoning to make decisions about object sorting based on descriptions
    and physical intuition, but it completely ignores reward feedback.
    
    Key characteristics:
    - Uses natural language descriptions to reason about objects
    - Applies common sense about pot sizes and object weights  
    - Never updates its behavior based on rewards (static)
    - Falls back to simulated reasoning if no API key available
    """
    
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize the LLM agent.
        
        Args:
            api_key: OpenAI API key (if None, uses simulated reasoning)
            model: Which OpenAI model to use for reasoning
        """
        api_key = api_key or os.getenv('OPENAI_API_KEY')  # Try environment variable
        self.use_real_api = api_key is not None          # Can we actually call OpenAI?
        
        if self.use_real_api:
            self.client = OpenAI(api_key=api_key)  # Initialize OpenAI client
            self.model = model                     # Store model name
        
        # System prompt that defines the LLM's task and reasoning framework
        # This gives the LLM context about the physical constraints and decision criteria
        self.system_prompt = """You are sorting objects into buckets based on their weight.

You have three buckets:
- Bucket A: A small 10-inch pot (for lighter objects)
- Bucket B: A large 30-inch pot (for heavier objects)
- Bucket C: Labeled "I don't know" (for uncertain cases)

Based on the object description and the bucket sizes, choose the most appropriate bucket.
Respond with a JSON object containing:
- "reasoning": Your thought process (1-2 sentences)
- "bucket": Your choice (A, B, or C)

Remember: Small pot for light things, big pot for heavy things."""
    
    def act(self, obj: Object, training: bool = False) -> str:
        """
        Choose bucket based on reasoning about object and pot sizes.
        
        The LLM uses natural language reasoning to decide which bucket is appropriate.
        Unlike the RL agent, it doesn't explore or change behavior based on rewards.
        The training parameter is ignored since this agent doesn't learn.
        
        Args:
            obj: Object to sort (uses description for reasoning)
            training: Ignored (LLM doesn't learn from rewards)
            
        Returns:
            Bucket choice: 'A', 'B', or 'C'
        """
        if self.use_real_api:
            return self._act_with_api(obj)      # Use real OpenAI API for reasoning
        else:
            return self._act_simulated(obj)     # Use hardcoded heuristics
    
    def _act_with_api(self, obj: Object) -> str:
        """
        Use real OpenAI API to make reasoning-based decisions.
        
        This sends the object description to GPT and asks it to reason
        about which bucket is most appropriate given the pot sizes.
        
        Args:
            obj: Object to sort
            
        Returns:
            Bucket choice from LLM reasoning
        """
        # Create a prompt with object information for the LLM to reason about
        prompt = f"""I need to sort this object:
Name: {obj.name}
Description: {obj.description}

Which bucket should I use?"""
        
        try:
            # Call OpenAI API with system prompt and object description
            response = self.client.chat.completions.create(
                model=self.model,                              # Use specified model (e.g., gpt-3.5-turbo)
                messages=[
                    {"role": "system", "content": self.system_prompt},  # Give context about task
                    {"role": "user", "content": prompt}                 # Ask about specific object
                ],
                temperature=0.3,                               # Low temperature for consistent reasoning
                max_tokens=100,                                # Don't need long responses
                response_format={"type": "json_object"}         # Force JSON output format
            )
            
            # Parse the JSON response to extract bucket choice
            content = response.choices[0].message.content
            result = json.loads(content)
            bucket = result.get('bucket', 'C').upper()         # Extract bucket, default to C
            
            # Validate the bucket choice
            if bucket in ['A', 'B', 'C']:
                return bucket
            else:
                return 'C'  # Default to safe choice if invalid response
                
        except Exception as e:
            print(f"API error: {e}")
            return 'C'  # Default to safe choice if API fails
    
    def _act_simulated(self, obj: Object) -> str:
        """
        Simulated LLM reasoning based on description keywords.
        
        When no API key is available, this method mimics how an LLM might
        reason about objects using simple keyword matching on descriptions.
        It demonstrates the kind of reasoning an LLM does but without
        the complexity of actual language model inference.
        
        Args:
            obj: Object to sort
            
        Returns:
            Bucket choice based on description keywords
        """
        # Convert description to lowercase for easier keyword matching
        desc_lower = obj.description.lower()
        
        # Look for keywords that suggest very light objects → bucket A
        if any(word in desc_lower for word in ["lightweight", "small", "easy to lift", "feather", "paper"]):
            return 'A'  # Clearly light objects go in small pot
        
        # Look for keywords that suggest very heavy objects → bucket B
        elif any(word in desc_lower for word in ["very heavy", "difficult", "concrete", "battery"]):
            return 'B'  # Clearly heavy objects go in large pot
        
        # Look for keywords that suggest medium weight → uncertain (bucket C)
        elif any(word in desc_lower for word in ["medium", "moderate"]):
            return 'C'  # LLM is uncertain about medium weights without exact threshold
        
        else:
            # Default reasoning: if "heavy" mentioned, use big pot, otherwise uncertain
            if "heavy" in desc_lower:
                return 'B'  # Generic "heavy" goes to large pot
            else:
                return 'C'  # When unsure, be safe
    
    def update(self, obj: Object, action: str, reward: float):
        """
        Static LLM doesn't learn from rewards.
        
        This is the key limitation: the LLM can reason but cannot improve
        its decision-making based on feedback. It will make the same mistakes
        repeatedly because it doesn't incorporate reward information.
        
        Args:
            obj: The object that was sorted (ignored)
            action: The action that was taken (ignored) 
            reward: The reward received (ignored)
        """
        pass  # Ignores all feedback! This is why it's "static"


class LLMWithRewardModel:
    """
    LLM agent that uses both reasoning and learns from rewards.
    
    This is the most sophisticated agent, combining the best of both worlds:
    - Starts with LLM reasoning (like StaticLLMAgent) for sensible initial behavior
    - Learns from rewards (like PureRLAgent) to discover the exact weight threshold
    - Maintains uncertainty estimates and explores intelligently near boundaries
    
    Key characteristics:
    - Uses LLM for initial reasonable guesses (avoids catastrophic failures)
    - Learns weight boundary estimates from reward feedback
    - Reduces uncertainty over time as it gathers more data
    - Explores strategically near the learned boundaries
    """
    
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize the hybrid LLM + Reward Learning agent.
        
        Args:
            api_key: OpenAI API key for LLM reasoning
            model: OpenAI model to use for reasoning
        """
        # Initialize LLM component for reasoning about object descriptions
        self.llm_agent = StaticLLMAgent(api_key, model)
        
        # Initialize reward learning component for discovering weight boundaries
        self.reward_memory = deque(maxlen=1000)  # Store recent (weight, action, reward) tuples
        
        # Maintain estimates of weight boundaries (learned from rewards)
        self.weight_estimates = {
            'A_upper': 3.0,   # Initial guess: objects <= 3kg probably go in bucket A  
            'B_lower': 27.0   # Initial guess: objects >= 27kg probably go in bucket B
        }
        # Note: Gap between 3-27kg represents uncertainty zone where LLM reasoning is used
        
        # Exploration parameters for strategic boundary exploration
        self.exploration_rate = 0.3      # 30% chance to explore near boundaries
        self.exploration_decay = 0.99    # Gradually reduce exploration as we learn
        self.exploration_min = 0.05      # Always maintain 5% exploration
        
        # Track learned threshold over time (for plotting)
        self.threshold_history = []
        
    def act(self, obj: Object, training: bool = True) -> str:
        """
        Choose bucket using LLM reasoning enhanced with learned knowledge.
        
        This method implements a sophisticated decision strategy that combines:
        1. LLM reasoning for initial sensible behavior
        2. Learned weight boundaries from reward feedback
        3. Strategic exploration near uncertain regions
        
        Args:
            obj: Object to sort
            training: Whether to use exploration (True) or pure exploitation (False)
            
        Returns:
            Bucket choice: 'A', 'B', or 'C'
        """
        # First, get the LLM's suggestion based on object description
        # This provides a reasonable fallback and handles edge cases well
        llm_choice = self.llm_agent.act(obj)
        
        # If not training (testing/deployment), use learned knowledge without exploration
        if not training:
            # Apply learned weight boundaries with confidence
            if obj.weight < self.weight_estimates['A_upper']:
                return 'A'  # Confident this is light enough for bucket A
            elif obj.weight > self.weight_estimates['B_lower']:
                return 'B'  # Confident this is heavy enough for bucket B
            else:
                return llm_choice  # Fall back to LLM reasoning in uncertain zone
        
        # During training, balance exploitation with strategic exploration
        if random.random() < self.exploration_rate:
            # STRATEGIC EXPLORATION: Focus exploration near learned boundaries
            # This is smarter than random exploration - we explore where we're most uncertain
            
            if abs(obj.weight - self.weight_estimates['A_upper']) < 3:
                # Object is near the A/B boundary - explore to refine this boundary
                return random.choice(['A', 'C'])  # Try A or play it safe with C
            elif abs(obj.weight - self.weight_estimates['B_lower']) < 3:
                # Object is near the B boundary - explore to refine this boundary  
                return random.choice(['B', 'C'])  # Try B or play it safe with C
            else:
                # Object is far from boundaries - use LLM reasoning
                return llm_choice
        else:
            # EXPLOITATION: Use learned knowledge with confidence margins
            
            if obj.weight < self.weight_estimates['A_upper'] - 1:
                # Well below A boundary - confidently use bucket A
                return 'A'
            elif obj.weight > self.weight_estimates['B_lower'] + 1:
                # Well above B boundary - confidently use bucket B
                return 'B'
            else:
                # In uncertain zone - rely on LLM reasoning
                return llm_choice
    
    def update(self, obj: Object, action: str, reward: float):
        """
        Update weight estimates based on rewards using boundary learning.
        
        This implements a form of online learning that maintains estimates of
        the weight boundaries between buckets. The key insight is that:
        - When A gets positive reward, we can increase A's upper weight limit
        - When A gets negative reward, we must decrease A's upper weight limit
        - When B gets positive reward, we can decrease B's lower weight limit  
        - When B gets negative reward, we must increase B's lower weight limit
        
        Args:
            obj: The object that was sorted
            action: The action taken ('A', 'B', or 'C')
            reward: The reward received (+1, -1, or 0)
        """
        # Store this experience for potential future analysis
        self.reward_memory.append((obj.weight, action, reward))
        
        # Update boundary estimates based on reward feedback
        if action == 'A':
            if reward > 0:
                # SUCCESS: Object was correctly placed in A (it was light enough)
                # This means our A_upper bound might be too conservative
                # We can safely increase it to at least this object's weight
                self.weight_estimates['A_upper'] = max(
                    self.weight_estimates['A_upper'],
                    obj.weight + 0.5  # Add small buffer for confidence
                )
            else:
                # FAILURE: Object was incorrectly placed in A (it was too heavy)
                # This means our A_upper bound is too liberal
                # We must decrease it to below this object's weight
                self.weight_estimates['A_upper'] = min(
                    self.weight_estimates['A_upper'],
                    obj.weight - 0.5  # Subtract buffer to avoid same mistake
                )
        
        elif action == 'B':
            if reward > 0:
                # SUCCESS: Object was correctly placed in B (it was heavy enough)
                # This means our B_lower bound might be too conservative
                # We can safely decrease it to at most this object's weight
                self.weight_estimates['B_lower'] = min(
                    self.weight_estimates['B_lower'],
                    obj.weight - 0.5  # Subtract buffer for confidence
                )
            else:
                # FAILURE: Object was incorrectly placed in B (it was too light)
                # This means our B_lower bound is too liberal
                # We must increase it to above this object's weight
                self.weight_estimates['B_lower'] = max(
                    self.weight_estimates['B_lower'],
                    obj.weight + 0.5  # Add buffer to avoid same mistake
                )
        
        # Note: Action 'C' provides no boundary information since it gets 0 reward
        # regardless of whether it was right or wrong
        
        # Track learned threshold (midpoint between boundaries) for plotting
        estimated_threshold = (self.weight_estimates['A_upper'] + self.weight_estimates['B_lower']) / 2.0
        self.threshold_history.append(estimated_threshold)
        
        # Gradually reduce exploration as we gain confidence in our estimates
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay


def train_agents(n_episodes: int = 500) -> Dict:
    """
    Train all three agents and compare their learning strategies.
    
    This function orchestrates the main experiment, training three different
    types of agents on the same task to demonstrate their different approaches
    to learning and decision-making.
    
    Args:
        n_episodes: Number of training episodes (objects to sort)
        
    Returns:
        Tuple of (trained_agents_dict, training_history_dict)
    """
    print("🏋️ OBJECT SORTING TASK")
    print("=" * 60)
    print("Training three agents to sort objects by weight...")
    print(f"Hidden weight threshold: 10kg")
    print("Agents don't know this threshold and must learn!\n")
    
    # Initialize environment and agents
    env = SortingEnvironment(weight_threshold=10.0)
    
    agents = {
        'Pure RL': PureRLAgent(),
        'Static LLM': StaticLLMAgent(),
        'LLM + Reward': LLMWithRewardModel()
    }
    
    # Track performance including threshold learning
    history = {name: {'rewards': [], 'buckets': defaultdict(int), 'thresholds': []} 
              for name in agents.keys()}
    
    # Training
    for episode in range(n_episodes):
        # Generate object
        obj = env.generate_object()
        
        # Print detailed info for episode 1
        if episode == 0:
            print(f"\n📍 EPISODE 1 BREAKDOWN:")
            print(f"Object: {obj}")
            print(f"Description: {obj.description}")
            print(f"True threshold: {env.weight_threshold}kg")
            print(f"Correct bucket: {'A' if obj.weight < env.weight_threshold else 'B'}")
            print()
        
        # Each agent sorts the object
        for name, agent in agents.items():
            # Act
            bucket = agent.act(obj, training=True)
            
            # Get reward
            reward = env.get_reward(obj, bucket)
            
            # Print episode 1 details
            if episode == 0:
                if hasattr(agent, 'get_learned_threshold'):
                    threshold = agent.get_learned_threshold()
                elif hasattr(agent, 'weight_estimates'):
                    threshold = (agent.weight_estimates['A_upper'] + agent.weight_estimates['B_lower']) / 2.0
                else:
                    threshold = "N/A (no learning)"
                    
                print(f"  {name:12}: bucket={bucket}, reward={reward:+.0f}, threshold={threshold}")
            
            # Update agent
            agent.update(obj, bucket, reward)
            
            # Track performance
            history[name]['rewards'].append(reward)
            history[name]['buckets'][bucket] += 1
            
            # Track thresholds for learning agents
            if hasattr(agent, 'threshold_history') and agent.threshold_history:
                history[name]['thresholds'].append(agent.threshold_history[-1])
            else:
                history[name]['thresholds'].append(None)  # Static LLM doesn't learn
        
        # Progress update
        if (episode + 1) % 100 == 0:
            print(f"\nEpisode {episode + 1}:")
            for name in agents.keys():
                recent_rewards = history[name]['rewards'][-100:]
                avg_reward = np.mean(recent_rewards)
                print(f"  {name:12} → Avg reward: {avg_reward:>5.2f}")
    
    return agents, history


def test_agents(agents: Dict, n_tests: int = 1000) -> Dict:
    """
    Test trained agents on new objects to evaluate final performance.
    
    This evaluates how well each agent performs after training by testing
    them on completely new objects. This tests generalization ability.
    
    Args:
        agents: Dictionary of trained agents
        n_tests: Number of test objects to generate
        
    Returns:
        Dictionary with detailed performance metrics for each agent
    """
    print("\n\n🎯 FINAL TEST (1000 objects)")
    print("=" * 60)
    
    env = SortingEnvironment(weight_threshold=10.0)
    results = {name: {'total_reward': 0, 'correct': 0, 'incorrect': 0, 'uncertain': 0}
              for name in agents.keys()}
    
    # Test each agent
    for _ in range(n_tests):
        obj = env.generate_object()
        
        for name, agent in agents.items():
            bucket = agent.act(obj, training=False)
            reward = env.get_reward(obj, bucket)
            
            results[name]['total_reward'] += reward
            
            if bucket == 'C':
                results[name]['uncertain'] += 1
            elif reward > 0:
                results[name]['correct'] += 1
            else:
                results[name]['incorrect'] += 1
    
    # Print results
    print("\nFINAL SCORES:")
    print("-" * 60)
    
    for name, res in results.items():
        total = res['total_reward']
        accuracy = res['correct'] / (res['correct'] + res['incorrect']) * 100 if (res['correct'] + res['incorrect']) > 0 else 0
        
        print(f"\n{name}:")
        print(f"  Total reward: {total:>4} points")
        print(f"  Correct:      {res['correct']:>4} ({res['correct']/10:.1f}%)")
        print(f"  Incorrect:    {res['incorrect']:>4} ({res['incorrect']/10:.1f}%)")
        print(f"  Uncertain:    {res['uncertain']:>4} ({res['uncertain']/10:.1f}%)")
        print(f"  Accuracy:     {accuracy:.1f}% (when not using C)")
    
    return results


def visualize_results(history: Dict, test_results: Dict):
    """Create visualizations including learned thresholds"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))  # Changed to 2x3 to fit threshold plot
    
    # 1. Learning curves
    ax1 = axes[0, 0]
    window = 50
    
    for name, data in history.items():
        rewards = data['rewards']
        avg_rewards = [np.mean(rewards[max(0, i-window+1):i+1]) 
                      for i in range(len(rewards))]
        ax1.plot(avg_rewards, label=name, linewidth=2)
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Learning Curves (50-episode average)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 2. Final test performance
    ax2 = axes[0, 1]
    names = list(test_results.keys())
    scores = [test_results[name]['total_reward'] for name in names]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    bars = ax2.bar(names, scores, color=colors)
    ax2.set_ylabel('Total Reward (1000 objects)')
    ax2.set_title('Final Test Performance')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, score in zip(bars, scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{score}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Learned Thresholds Over Time
    ax3 = axes[0, 2]
    
    # Plot threshold evolution for learning agents
    for i, (name, data) in enumerate(history.items()):
        thresholds = data['thresholds']
        if any(t is not None for t in thresholds):  # Only plot if agent learns
            # Filter out None values and create corresponding episode indices
            valid_thresholds = [(i, t) for i, t in enumerate(thresholds) if t is not None]
            if valid_thresholds:
                episodes, threshold_values = zip(*valid_thresholds)
                ax3.plot(episodes, threshold_values, label=f"{name}", 
                        linewidth=2, color=colors[i])
    
    # Add true threshold line
    ax3.axhline(y=10.0, color='red', linestyle='--', linewidth=2, 
               label='True Threshold (10kg)', alpha=0.8)
    
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Learned Threshold (kg)')
    ax3.set_title('Threshold Learning Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 25)  # Reasonable range for thresholds
    
    # 4. Bucket usage over time
    ax4 = axes[1, 0]
    
    # Calculate bucket usage in windows
    window_size = 100
    n_windows = len(history['Pure RL']['rewards']) // window_size
    
    for i, (name, data) in enumerate(history.items()):
        c_usage = []
        rewards = data['rewards']
        
        for w in range(n_windows):
            window_rewards = rewards[w*window_size:(w+1)*window_size]
            c_percentage = window_rewards.count(0) / len(window_rewards) * 100
            c_usage.append(c_percentage)
        
        ax4.plot(range(n_windows), c_usage, label=f"{name} (C usage)", 
                linewidth=2, color=colors[i])
    
    ax4.set_xlabel('Training Progress (hundreds of episodes)')
    ax4.set_ylabel('% Using "I don\'t know" bucket')
    ax4.set_title('Confidence Over Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Accuracy comparison
    ax5 = axes[1, 1]
    
    accuracies = []
    uncertainties = []
    
    for name in names:
        res = test_results[name]
        correct = res['correct']
        incorrect = res['incorrect']
        uncertain = res['uncertain']
        
        if correct + incorrect > 0:
            accuracy = correct / (correct + incorrect) * 100
        else:
            accuracy = 0
            
        accuracies.append(accuracy)
        uncertainties.append(uncertain / 10)  # Percentage
    
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = ax5.bar(x - width/2, accuracies, width, label='Accuracy (when decided)', color='green')
    bars2 = ax5.bar(x + width/2, uncertainties, width, label='% Uncertain', color='orange')
    
    ax5.set_xlabel('Agent Type')
    ax5.set_ylabel('Percentage')
    ax5.set_title('Decision Quality')
    ax5.set_xticks(x)
    ax5.set_xticklabels(names)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Final Threshold Comparison
    ax6 = axes[1, 2]
    
    # Show final learned thresholds vs true threshold
    final_thresholds = []
    agent_names = []
    
    for name in names:
        if history[name]['thresholds'] and any(t is not None for t in history[name]['thresholds']):
            # Get the last non-None threshold
            final_threshold = None
            for t in reversed(history[name]['thresholds']):
                if t is not None:
                    final_threshold = t
                    break
            if final_threshold is not None:
                final_thresholds.append(final_threshold)
                agent_names.append(name)
    
    if final_thresholds:
        bars = ax6.bar(agent_names, final_thresholds, color=[colors[names.index(name)] for name in agent_names])
        ax6.axhline(y=10.0, color='red', linestyle='--', linewidth=2, alpha=0.8, label='True Threshold')
        
        # Add value labels on bars
        for bar, threshold in zip(bars, final_thresholds):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{threshold:.1f}kg', ha='center', va='bottom', fontweight='bold')
        
        ax6.set_ylabel('Learned Threshold (kg)')
        ax6.set_title('Final Learned Thresholds')
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')
        ax6.set_ylim(0, max(max(final_thresholds), 15))
    
    plt.suptitle('Object Sorting Task: Comparing RL, LLM, and LLM+Reward', fontsize=16)
    plt.tight_layout()
    plt.savefig('sorting_task_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def explain_results():
    """Explain what happened and why"""
    print("\n\n💡 KEY INSIGHTS")
    print("=" * 60)
    
    print("""
1. PURE RL AGENT:
   - Started completely blind, trying random buckets
   - Gradually learned the ~10kg threshold through trial and error
   - Eventually achieves high accuracy but took many mistakes to learn
   - Never uses the "I don't know" bucket once confident

2. STATIC LLM AGENT:
   - Uses reasoning about pot sizes (10" vs 30") and object descriptions
   - Makes reasonable guesses from the start (no catastrophic failures)
   - But can't learn the exact 10kg threshold from feedback
   - Often uses "I don't know" for medium-weight objects
   - Plateaus at suboptimal performance

3. LLM + REWARD MODEL:
   - Starts with LLM's reasonable guesses (safe from the beginning)
   - Learns the exact weight threshold from rewards
   - Reduces uncertainty over time as it learns
   - Achieves best final performance

WHY THIS MATTERS:
- Pure RL: Like deploying a blind robot - eventually optimal but dangerous start
- Static LLM: Like a smart human who can't learn from experience
- LLM + Reward: Best of both worlds - starts smart and gets smarter!
""")


if __name__ == "__main__":
    # Train agents
    trained_agents, training_history = train_agents(n_episodes=1000)
    
    # Test agents
    test_results = test_agents(trained_agents, n_tests=100)
    
    # Visualize
    visualize_results(training_history, test_results)
    
    # Explain
    explain_results()
    
    print("\n✅ Experiment complete! Check 'sorting_task_results.png' for visualizations.")
