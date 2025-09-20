# Go Playing AI Agent

> **An AI system that masters the game of Go on a 5x5 board**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![AI Agent](https://img.shields.io/badge/AI-Game%20Agent-green.svg)](https://github.com/Nikhil-Singla/go-playing-agent)
[![Algorithm](https://img.shields.io/badge/Algorithm-Minimax%20%2B%20Alpha--Beta-red.svg)](https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning)

## 🏆 **Project Highlights**
#### **About This Project**

This project demonstrates **advanced AI programming skills** through a complete implementation of a Go-playing agent that competes intelligently on a 5×5 board. The AI combines classical game theory algorithms with sophisticated heuristic evaluation to make smart decisions in real-time.

#### **What is Go?**
Go (Weiqi/Baduk) is one of the world's most complex strategy games, famously more challenging than chess. Players alternate placing stones to control territory, with simple rules but profound strategic depth.

#### **Strategy Features**
- **Tactical Awareness**: Prioritizes tactical objectives (captures/threats)
- **Multi-Phase Gameplay**: Adapts based on stage - opening/midgame/endgame
- **Sophisticated Heuristics**: Evaluates position quality holistically
- **Game Theory**: Implements optimal decision-making

## 🖥️ **Technical Architecture**

#### **Algorithm Implementation**
- **Minimax with Alpha-Beta Pruning**: Reduces search complexity from O(b^d) to O(b^(d/2))
- **Dynamic Depth Control**: Adjusts search depth based on game phase and remaining moves
- **Heuristic Evaluation**: Multi-layered position assessment considering:
  - Material balance 
  - Positional value
  - Stone connectivity 
  - Shape quality  
  - Territory control 
  - Eye formation
  - Liberty count 
  - Group safety

#### **Game Logic Mastery**
- **Complete Go Rules Implementation**: Liberty checking, capture resolution, Ko rule
- **Strategic Pattern Recognition**: Identifies and evaluates Go-specific formations
- **Board State Management**: Efficient game state tracking and move validation

#### **Performance Optimization**
- **Targetted Improvement**: Code debugged and optimized using the cProfiler module for better performance
- **Smart Move Ordering**: Prioritizes center positions for better alpha-beta pruning
- **Capture Priority**: Immediate tactical evaluation for quick wins

## 📈 **Performance Characteristics**

| Metric | Performance |
|--------|-------------|
| **Search Depth** | 3-4 moves ahead (adaptive) |
| **Move Generation** | < 1 second per move |
| **Strategic Phases** | 3 distinct playing styles |
| **Rule Compliance** | 100% Go rule adherence |
| **Code Quality** | Extensively documented with AI assistance |

- Note: Copilot is used ONLY for DOCUMENTATION purposes. The rest of the code is self-written and implemented.
- The original code can be viewed by going to the first commit of the file.

## 🔧 **Skills Overview**

- **Algorithms & AI**: Minimax, Alpha-Beta Pruning, graph-based group analysis, game theory, heuristic evaluation, strategic planning, reinforcement learning, deep learning, neural networks, game AI  
- **Software Engineering**: Python, clean code, debugging, optimization, file I/O, documentation, academic project design, web hosting compatibility  
- **Core Strengths**: Strategic thinking, problem solving, AI for strategic games (Go, 5x5 board focus), academic research applications

## 🎯 **Usage**

### **Quick Start**
```bash
# Clone the repository
git clone https://github.com/Nikhil-Singla/go-playing-agent.git
cd go-playing-agent

# Open self hosting/ Go to the website 
🚧 **TODO**: This section is under construction. Content will be added soon.
```


## 🚀 **Future Enhancements**

- [ ] **Neural Network Integration**: Replace heuristics with learned evaluation
- [ ] **Monte Carlo Tree Search**: Implement modern Go AI techniques
- [ ] **Larger Board Support**: Scale to 9×9, 13×13, and 19×19 boards
- [ ] **Web Interface**: Browser-based gameplay and visualization

## 🤝 **Connect With Me**

**Nikhil Singla** - AI/Software Engineer

- 💼 **LinkedIn**: [Connect for professional opportunities](https://linkedin.com/in/nikhil-singla)
- 📧 **Email**: [Reach out for collaborations](mailto:nsingla3.14@gmail.com)
- 💻 **GitHub**: [@Nikhil-Singla](https://github.com/Nikhil-Singla)

## 🎮 Go Game Rules (5x5 Board)

Go is an ancient strategy game adapted here for a 5x5 board for faster experimentation:
- **Objective**: Control more stones than your opponent on the board
- **5x5 Advantage**: Faster games ideal for AI training and analysis
- **Placement**: Players alternately place stones on intersections
- **Capture**: Surround opponent stones to remove them
- **Ko Rule**: Prevents immediate recapture situations
- **Scoring**: Total stones value

## 📚 References

- [Monte Carlo Tree Search](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)
- [Go Rules and Strategy](https://en.wikipedia.org/wiki/Go_(game))
- [AlphaGo Paper](https://www.nature.com/articles/nature16961)

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

