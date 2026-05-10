/**
 * Mohammed IFKIRNE
 */

/**
 * Implementation de Q-Learning (Apprentissage par Renforcement)
 *
 * L'apprentissage par renforcement permet a un agent d'apprendre comment agir
 * dans un environnement pour maximiser ses recompenses. Q-Learning est un
 * algorithme "model-free" qui apprend la valeur d'une action dans un etat donne
 * sans connaitre le modele complet de l'environnement au prealable.
 *
 * Concepts mathematiques cles:
 * 1. Etat (State) S : La situation actuelle de l'agent (ex: position sur une
 * grille).
 * 2. Action A : Choix effectue par l'agent (ex: Haut, Bas, Gauche, Droite).
 * 3. Recompense (Reward) R : Feedback de l'environnement apres une action.
 * 4. Q-Table : Tableau Q(s, a) stockant la recompense totale attendue.
 *
 * Equation de Bellman (Mise a jour de Q) :
 * Q(s, a) = Q(s, a) + alpha * [ R + gamma * max(Q(s', a')) - Q(s, a) ]
 *
 * Ou :
 * - alpha (Learning Rate) : Vitesse d'apprentissage (0 = rien n'est appris).
 * - gamma (Discount Factor) : Importance des recompenses futures (0 = myope, 1
 * = long terme).
 * - max(Q(s', a')) : Meilleure valeur possible depuis le nouvel etat s'.
 *
 * Exploration vs Exploitation (Epsilon-Greedy) :
 * L'agent choisit une action au hasard avec une probabilite Epsilon pour
 * explorer, ou choisit la meilleure action connue (Exploitation) avec
 * probabilite 1 - Epsilon.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Hyperparametres
#define ALPHA 0.1f    // Taux d'apprentissage
#define GAMMA 0.9f    // Facteur d'actualisation (Discount)
#define EPSILON 0.2f  // Probabilite d'exploration
#define EPISODES 1000 // Nombre de parties a jouer
#define MAX_STEPS 100 // Nombre max de pas par episode avant d'abandonner

// Parametres de l'environnement (Grille 4x4)
#define GRID_SIZE 4
#define NUM_STATES (GRID_SIZE * GRID_SIZE)
#define NUM_ACTIONS 4

// Actions possibles
#define UP 0
#define RIGHT 1
#define DOWN 2
#define LEFT 3

// Recompenses et etats speciaux
#define GOAL_STATE 15 // Position (3,3)
#define TRAP_STATE 5  // Position (1,1)

// Q-Table globale
float Q[NUM_STATES][NUM_ACTIONS];

// Initialise la Q-Table a 0
void init_q_table() {
  for (int s = 0; s < NUM_STATES; s++) {
    for (int a = 0; a < NUM_ACTIONS; a++) {
      Q[s][a] = 0.0f;
    }
  }
}

// Renvoie une valeur aleatoire entre 0 et 1
float rand_float() { return (float)rand() / (float)RAND_MAX; }

// Strategie Epsilon-Greedy pour choisir une action
int choose_action(int state) {
  // Exploration : choix aleatoire
  if (rand_float() < EPSILON) {
    return rand() % NUM_ACTIONS;
  }

  // Exploitation : choisir la meilleure action connue
  int best_action = 0;
  float best_q = Q[state][0];
  for (int a = 1; a < NUM_ACTIONS; a++) {
    if (Q[state][a] > best_q) {
      best_q = Q[state][a];
      best_action = a;
    }
  }
  return best_action;
}

// Execute une action dans l'environnement et retourne le nouvel etat et la
// recompense
int step(int state, int action, float *reward, int *done) {
  int x = state % GRID_SIZE;
  int y = state / GRID_SIZE;

  // Calcul du nouvel etat
  if (action == UP && y > 0)
    y--;
  else if (action == DOWN && y < GRID_SIZE - 1)
    y++;
  else if (action == LEFT && x > 0)
    x--;
  else if (action == RIGHT && x < GRID_SIZE - 1)
    x++;

  int next_state = y * GRID_SIZE + x;

  // Evaluation de la recompense et de l'etat terminal
  *done = 0;
  if (next_state == GOAL_STATE) {
    *reward = 10.0f; // Succes !
    *done = 1;
  } else if (next_state == TRAP_STATE) {
    *reward = -10.0f; // Piege !
    *done = 1;
  } else if (next_state == state) {
    *reward = -0.5f; // Penalite pour avoir tape un mur
  } else {
    *reward = -0.1f; // Petite penalite a chaque pas pour encourager le chemin
                     // le plus court
  }

  return next_state;
}

// Renvoie la valeur Q maximale pour un etat donne : max(Q(s', a'))
float get_max_q(int state) {
  float max_q = Q[state][0];
  for (int a = 1; a < NUM_ACTIONS; a++) {
    if (Q[state][a] > max_q) {
      max_q = Q[state][a];
    }
  }
  return max_q;
}

// Affichage visuel du meilleur chemin appris
void print_policy() {
  printf("\nMeilleure politique apprise :\n");
  for (int y = 0; y < GRID_SIZE; y++) {
    for (int x = 0; x < GRID_SIZE; x++) {
      int state = y * GRID_SIZE + x;

      if (state == GOAL_STATE) {
        printf(" [ G ] "); // Goal
      } else if (state == TRAP_STATE) {
        printf(" [ X ] "); // Trap
      } else {
        int best_a = 0;
        float best_q = Q[state][0];
        for (int a = 1; a < NUM_ACTIONS; a++) {
          if (Q[state][a] > best_q) {
            best_q = Q[state][a];
            best_a = a;
          }
        }

        // Affichage des fleches de direction
        if (best_a == UP)
          printf(" [ ^ ] ");
        else if (best_a == DOWN)
          printf(" [ v ] ");
        else if (best_a == LEFT)
          printf(" [ < ] ");
        else if (best_a == RIGHT)
          printf(" [ > ] ");
      }
    }
    printf("\n");
  }
}

int main() {
  srand((unsigned int)time(NULL));

  printf("--- Q-Learning (Apprentissage par Renforcement) ---\n");
  printf("Environnement : Grille 4x4\n");
  printf("Depart (0,0) en haut a gauche.\n");
  printf("Objectif 'G' (3,3) en bas a droite (Recompense: +10).\n");
  printf("Piege 'X' (1,1) au milieu (Recompense: -10).\n\n");

  init_q_table();

  // Entrainement
  printf("Entrainement de l'agent sur %d episodes...\n", EPISODES);

  for (int episode = 0; episode < EPISODES; episode++) {
    int state = 0; // Toujours commencer en haut a gauche (0,0)
    int done = 0;

    for (int step_idx = 0; step_idx < MAX_STEPS && !done; step_idx++) {
      // 1. Choisir une action (Strategie Epsilon-Greedy)
      int action = choose_action(state);

      // 2. Executer l'action dans l'environnement
      float reward;
      int next_state = step(state, action, &reward, &done);

      // 3. Equation de Bellman : Mettre a jour la Q-Table
      float max_future_q = get_max_q(next_state);
      float td_target = reward + GAMMA * max_future_q;
      float td_error = td_target - Q[state][action];

      Q[state][action] = Q[state][action] + ALPHA * td_error;

      // 4. Passer a l'etat suivant
      state = next_state;
    }
  }

  printf("Entrainement termine.\n");

  // Afficher la politique finale (le chemin optimal decouvert par l'agent)
  print_policy();

  return 0;
}

// Affichage:

// --- Q-Learning (Apprentissage par Renforcement) ---
// Environnement : Grille 4x4
// Depart (0,0) en haut a gauche.
// Objectif 'G' (3,3) en bas a droite (Recompense: +10).
// Piege 'X' (1,1) au milieu (Recompense: -10).

// Entrainement de l'agent sur 1000 episodes...
// Entrainement termine.

// Meilleure politique apprise :
//  [ > ]  [ > ]  [ v ]  [ v ]
//  [ ^ ]  [ X ]  [ v ]  [ v ]
//  [ v ]  [ v ]  [ > ]  [ v ]
//  [ > ]  [ > ]  [ > ]  [ G ]