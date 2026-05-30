/**
 * AI Monster Generator — Foundry VTT Module
 *
 * Adds two buttons to the Actors directory:
 *   "Generate Monster" — creates a single actor from a concept prompt
 *   "Build Encounter"  — searches the catalog and returns a matched encounter
 */

const MODULE_ID = "monster-generator";

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

Hooks.once("init", () => {
  game.settings.register(MODULE_ID, "apiUrl", {
    name: game.i18n.localize("MGEN.Settings.ApiUrl"),
    hint: game.i18n.localize("MGEN.Settings.ApiUrlHint"),
    scope: "world",
    config: true,
    type: String,
    default: "http://localhost:8765",
  });
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function apiUrl(path) {
  const base = game.settings.get(MODULE_ID, "apiUrl").replace(/\/$/, "");
  return `${base}${path}`;
}

async function apiFetch(path, body) {
  const resp = await fetch(apiUrl(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail || resp.statusText);
  }
  return resp.json();
}

// ---------------------------------------------------------------------------
// Generate Monster dialog
// ---------------------------------------------------------------------------

class GenerateMonsterDialog extends Dialog {
  constructor(options = {}) {
    const content = `
      <div id="monster-generator-dialog">
        <p>Describe a creature concept:</p>
        <input class="mgen-input" id="mgen-prompt" type="text"
               placeholder="${game.i18n.localize("MGEN.Placeholder")}" autofocus />
        <div class="mgen-status" id="mgen-status"></div>
      </div>`;

    super({
      title: game.i18n.localize("MGEN.Generate"),
      content,
      buttons: {
        generate: {
          icon: '<i class="fas fa-dragon"></i>',
          label: "Generate",
          callback: (html) => this._onGenerate(html),
        },
        cancel: { label: "Cancel" },
      },
      default: "generate",
      ...options,
    });
  }

  activateListeners(html) {
    super.activateListeners(html);
    html.find("#mgen-prompt").on("keydown", (e) => {
      if (e.key === "Enter") { e.preventDefault(); this._onGenerate(html); }
    });
  }

  async _onGenerate(html) {
    const prompt = html.find("#mgen-prompt").val().trim();
    if (!prompt) return;

    const status = html.find("#mgen-status");
    status.text(game.i18n.localize("MGEN.Generating")).removeClass("mgen-error");

    // Disable the button to prevent double-clicks
    html.find("button[data-button='generate']").prop("disabled", true);

    try {
      const data = await apiFetch("/generate", { prompt });
      const actor = await Actor.create(data.actor);
      ui.notifications.info(`Created: ${actor.name}`);
      actor.sheet.render(true);
      this.close();
    } catch (err) {
      status.text(`Error: ${err.message}`).addClass("mgen-error");
      html.find("button[data-button='generate']").prop("disabled", false);
    }
  }
}

// ---------------------------------------------------------------------------
// Encounter dialog
// ---------------------------------------------------------------------------

class EncounterDialog extends Dialog {
  constructor(options = {}) {
    const content = `
      <div id="monster-generator-dialog">
        <p>Describe the encounter:</p>
        <input class="mgen-input" id="mgen-enc-prompt" type="text"
               placeholder="${game.i18n.localize("MGEN.EncounterPlaceholder")}" autofocus />
        <div class="mgen-status" id="mgen-enc-status"></div>
        <div id="mgen-enc-result"></div>
      </div>`;

    super({
      title: game.i18n.localize("MGEN.Encounter"),
      content,
      buttons: {
        build: {
          icon: '<i class="fas fa-users"></i>',
          label: "Build Encounter",
          callback: (html) => this._onBuild(html),
        },
        cancel: { label: "Cancel" },
      },
      default: "build",
      ...options,
    });
  }

  activateListeners(html) {
    super.activateListeners(html);
    html.find("#mgen-enc-prompt").on("keydown", (e) => {
      if (e.key === "Enter") { e.preventDefault(); this._onBuild(html); }
    });
  }

  async _onBuild(html) {
    const prompt = html.find("#mgen-enc-prompt").val().trim();
    if (!prompt) return;

    const status = html.find("#mgen-enc-status");
    const result = html.find("#mgen-enc-result");
    status.text(game.i18n.localize("MGEN.Generating")).removeClass("mgen-error");
    result.empty();
    html.find("button[data-button='build']").prop("disabled", true);

    try {
      const enc = await apiFetch("/encounter", { prompt });

      // Render encounter summary
      const rows = enc.monsters.map(m =>
        `<div class="mgen-monster-row">
           <span class="mgen-monster-name">${m.name} ${m.count > 1 ? "×" + m.count : ""}</span>
           <span class="mgen-monster-meta">CR ${m.cr} | HP ${m.hp} | AC ${m.ac} | ${m.role}</span>
         </div>`
      ).join("");

      result.html(`
        <div class="mgen-encounter-result">
          <h3>${enc.difficulty.toUpperCase()} — Adjusted XP: ${enc.adjusted_xp.toLocaleString()}</h3>
          ${rows}
          <p style="margin-top:8px;font-style:italic">${enc.gm_note}</p>
        </div>`);

      status.text("");
      html.find("button[data-button='build']").prop("disabled", false);
    } catch (err) {
      status.text(`Error: ${err.message}`).addClass("mgen-error");
      html.find("button[data-button='build']").prop("disabled", false);
    }
  }
}

// ---------------------------------------------------------------------------
// Add buttons to the Actors directory
// ---------------------------------------------------------------------------

Hooks.on("renderActorDirectory", (app, html) => {
  const header = html.find(".directory-header .header-actions");

  const genBtn = $(
    `<button class="mgen-btn" title="${game.i18n.localize("MGEN.Generate")}">
       <i class="fas fa-dragon"></i> AI Monster
     </button>`
  );
  genBtn.on("click", () => new GenerateMonsterDialog().render(true));

  const encBtn = $(
    `<button class="mgen-btn" title="${game.i18n.localize("MGEN.Encounter")}">
       <i class="fas fa-users"></i> AI Encounter
     </button>`
  );
  encBtn.on("click", () => new EncounterDialog().render(true));

  header.prepend(encBtn).prepend(genBtn);
});
