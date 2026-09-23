"use strict";

const SUPABASE_URL = "https://odfwbtkfyhomrygpnmec.supabase.co";
const SUPABASE_PUBLISHABLE_KEY = "sb_publishable_2sjaqhhLe2LQ7ArmL8KYAQ_dzgm-Jt8";

const db = supabase.createClient(
  SUPABASE_URL,
  SUPABASE_PUBLISHABLE_KEY
);

const EXPERIMENT_VERSION = "triangle_preference_25x25_v2";
const GENERATED_TRIAL_ROOT = "generated_trials";
const TRIANGLE_TRIAL_COUNT = 30;
const PREFERENCE_TRIAL_COUNT = 30;

const AUDIO_LOAD_TIMEOUT_MS = 12000;
const AUDIO_RETRY_DELAY_MS = 1500;


const HEADPHONE_CHECKS = [
  {
    id: "hp_1",
    urls: [
      "screening/hp_1_a.wav",
      "screening/hp_1_b.wav",
      "screening/hp_1_c.wav"
    ],
    correctPosition: 2
  },
  {
    id: "hp_2",
    urls: [
      "screening/hp_2_a.wav",
      "screening/hp_2_b.wav",
      "screening/hp_2_c.wav"
    ],
    correctPosition: 3
  },
  {
    id: "hp_3",
    urls: [
      "screening/hp_3_a.wav",
      "screening/hp_3_b.wav",
      "screening/hp_3_c.wav"
    ],
    correctPosition: 1
  }
];


const stages = {
  consent: document.getElementById("consentStage"),
  headphone: document.getElementById("headphoneStage"),
  calibration: document.getElementById("calibrationStage"),
  instructions: document.getElementById("instructionsStage"),
  triangle: document.getElementById("triangleStage"),
  preference: document.getElementById("preferenceStage"),
  break: document.getElementById("breakStage"),
  done: document.getElementById("doneStage"),
  fail: document.getElementById("screenFailStage")
};


const triangleAudio = [
  document.getElementById("triangleAudio1"),
  document.getElementById("triangleAudio2"),
  document.getElementById("triangleAudio3")
];


const preferenceAudio = [
  document.getElementById("preferenceAudioA"),
  document.getElementById("preferenceAudioB")
];


const triangleButtons = Array.from(
  document.querySelectorAll(".triangleChoice")
);


const preferenceButtons = Array.from(
  document.querySelectorAll(".preferenceChoice")
);


let participantId = getOrCreateLocalId(
  "speech_study_participant_id"
);

let sessionId = crypto.randomUUID();

let environmentMetadata = {};
let participantMetadata = {};

let headphoneAnswers = {};
let headphonePlayCounts = {};
let headphonePassed = false;

let triangleTrials = [];
let preferenceTrials = [];

let scheduleId = null;
let scheduleCount = null;

let blockOrder = [];
let currentBlockPosition = 0;
let currentTask = null;

let triangleIndex = 0;
let preferenceIndex = 0;

let trialStartMs = 0;

let trianglePlayCounts = [0, 0, 0];
let triangleHasPlayed = [false, false, false];

let preferencePlayCounts = [0, 0];
let preferenceHasPlayed = [false, false];

let saving = false;



function getOrCreateLocalId(key) {
  let value = localStorage.getItem(key);

  if (!value) {
    value = crypto.randomUUID();
    localStorage.setItem(key, value);
  }

  return value;
}



function collectParticipantMetadata() {
  const hearingAnswer =
    document.getElementById("normalHearing")?.value || "";

  participantMetadata = {
    age_group:
      document.getElementById("ageGroup")?.value || null,

    sex_assigned_at_birth:
      document.getElementById("sexAssignedAtBirth")?.value || null,

    gender_identity:
      document.getElementById("genderIdentity")?.value.trim() || null,

    native_language:
      document.getElementById("nativeLanguage")?.value.trim() || null,

    english_proficiency:
      document.getElementById("englishProficiency")?.value || null,

    english_use_frequency:
      document.getElementById("englishUseFrequency")?.value || null,

    normal_hearing:
      hearingAnswer === "no"
        ? true
        : hearingAnswer === "yes"
          ? false
          : null,

    hearing_aid_use:
      document.getElementById("hearingAidUse")?.value || null,

    headphone_type:
      document.getElementById("deviceType")?.value || null,

    listening_environment:
      document.getElementById("listeningEnvironment")?.value || null,

    browser:
      navigator.userAgent,

    operating_system:
      navigator.platform
  };
}



function formatScheduleId(index) {
  return `set_${String(index).padStart(3, "0")}`;
}



async function loadScheduleManifest() {
  const response = await fetch(
    `${GENERATED_TRIAL_ROOT}/schedule_manifest.json`
      + `?v=${encodeURIComponent(EXPERIMENT_VERSION)}`,
    {
      cache: "no-store"
    }
  );

  if (!response.ok) {
    throw new Error(
      `Could not load the schedule manifest `
      + `(HTTP ${response.status}).`
    );
  }

  const manifest = await response.json();

  const count = Number(
    manifest.schedule_count
    ?? manifest.num_schedules
  );

  if (!Number.isInteger(count) || count < 1) {
    console.error(
      "Invalid schedule manifest:",
      manifest
    );

    throw new Error(
      "schedule_manifest.json must contain a positive "
      + "integer schedule_count."
    );
  }

  scheduleCount = count;

  return manifest;
}



async function claimParticipantSchedule() {
  if (scheduleId) {
    return scheduleId;
  }

  if (!scheduleCount) {
    await loadScheduleManifest();
  }

  const { data, error } = await db.rpc(
    "claim_listening_schedule",
    {
      p_participant_id: participantId,
      p_experiment_version: EXPERIMENT_VERSION,
      p_schedule_count: scheduleCount
    }
  );

  if (error) {
    console.error(
      "Could not claim schedule:",
      error
    );

    throw new Error(
      `Could not assign a listening schedule: `
      + error.message
    );
  }

  let rawScheduleNumber = data;

  if (Array.isArray(data)) {
    rawScheduleNumber = data[0];
  }

  if (
    rawScheduleNumber !== null
    && typeof rawScheduleNumber === "object"
  ) {
    rawScheduleNumber =
      rawScheduleNumber.schedule_number
      ?? rawScheduleNumber.claim_listening_schedule
      ?? rawScheduleNumber.schedule_id;
  }

  const scheduleNumber =
    Number(rawScheduleNumber);

  if (
    !Number.isInteger(scheduleNumber)
    || scheduleNumber < 1
    || scheduleNumber > scheduleCount
  ) {
    console.error(
      "Invalid schedule RPC result:",
      data
    );

    throw new Error(
      "Supabase returned an invalid schedule number."
    );
  }

  scheduleId =
    formatScheduleId(scheduleNumber);

  return scheduleId;
}



function showOnly(stage) {
  for (const element of Object.values(stages)) {
    element.hidden = true;
  }

  stage.hidden = false;

  window.scrollTo({
    top: 0,
    behavior: "smooth"
  });
}



function shuffle(values) {
  const result = [...values];

  for (
    let i = result.length - 1;
    i > 0;
    i -= 1
  ) {
    const j =
      Math.floor(
        Math.random() * (i + 1)
      );

    [
      result[i],
      result[j]
    ] = [
      result[j],
      result[i]
    ];
  }

  return result;
}



function chooseBlockOrder() {
  blockOrder =
    Math.random() < 0.5
      ? ["triangle", "preference"]
      : ["preference", "triangle"];
}



async function upsertSession(status) {
  const row = {
    participant_id:
      participantId,

    session_id:
      sessionId,

    experiment_version:
      EXPERIMENT_VERSION,

    schedule_id:
      scheduleId,

    status,

    consent_given:
      Boolean(
        environmentMetadata.consent_given
      ),

    adult_confirmed:
      Boolean(
        environmentMetadata.adult_confirmed
      ),

    quiet_room_confirmed:
      Boolean(
        environmentMetadata.quiet_room_confirmed
      ),

    headphones_confirmed:
      Boolean(
        environmentMetadata.headphones_confirmed
      ),

    device_type:
      environmentMetadata.device_type
      || null,

    environment_noise:
      environmentMetadata.environment_noise
      || null,

    age_group:
      participantMetadata.age_group,

    sex_assigned_at_birth:
      participantMetadata.sex_assigned_at_birth,

    gender_identity:
      participantMetadata.gender_identity
      ?? null,

    native_language:
      participantMetadata.native_language,

    english_proficiency:
      participantMetadata.english_proficiency,

    english_use_frequency:
      participantMetadata.english_use_frequency,

    normal_hearing:
      participantMetadata.normal_hearing,

    hearing_aid_use:
      participantMetadata.hearing_aid_use,

    headphone_type:
      participantMetadata.headphone_type,

    listening_environment:
      participantMetadata.listening_environment,

    browser:
      participantMetadata.browser,

    operating_system:
      participantMetadata.operating_system,

    headphone_passed:
      headphonePassed,

    headphone_correct:
      HEADPHONE_CHECKS.reduce(
        (sum, trial) => {
          return (
            sum
            + (
              headphoneAnswers[trial.id]
                === trial.correctPosition
                ? 1
                : 0
            )
          );
        },
        0
      ),

    headphone_total:
      HEADPHONE_CHECKS.length,

    block_order:
      blockOrder.join("_then_")
      || null,

    triangle_completed:
      triangleIndex,

    preference_completed:
      preferenceIndex,

    user_agent:
      navigator.userAgent
  };

  const { error } =
    await db
      .from("sessions")
      .upsert(
        row,
        {
          onConflict: "session_id"
        }
      );

  if (error) {
    console.error(
      "Could not save session:",
      error
    );

    throw new Error(
      error.message
    );
  }
}



/*
 * ------------------------------------------------------------
 * Robust audio loading
 * ------------------------------------------------------------
 *
 * The same audio URL is retried indefinitely.
 *
 * Nothing is skipped.
 * Nothing is replaced.
 * The trial index never changes because of an audio error.
 *
 * A retry is triggered when:
 *
 *   1. The browser emits an audio "error" event.
 *   2. Loading takes longer than AUDIO_LOAD_TIMEOUT_MS.
 *
 * After AUDIO_RETRY_DELAY_MS the exact same URL is loaded again.
 *
 * WeakMap is used so each <audio> element has its own load state.
 */

const audioLoadStates =
  new WeakMap();



function cancelPendingAudioLoad(audio) {
  const state =
    audioLoadStates.get(audio);

  if (!state) {
    return;
  }

  state.cancelled = true;

  if (state.timeoutId !== null) {
    clearTimeout(
      state.timeoutId
    );
  }

  if (state.retryId !== null) {
    clearTimeout(
      state.retryId
    );
  }

  if (state.controller) {
    state.controller.abort();
  }

  audioLoadStates.delete(audio);
}



function loadAudioWithRetry(
  audio,
  url
) {
  /*
   * Stop any previous retry process associated with
   * this specific <audio> element.
   */
  cancelPendingAudioLoad(audio);

  const state = {
    attempt: 0,
    timeoutId: null,
    retryId: null,
    controller: null,
    cancelled: false
  };

  audioLoadStates.set(
    audio,
    state
  );


  const tryLoad = () => {
    /*
     * This protects against a stale retry firing after
     * the participant has already moved to another trial.
     */
    if (
      state.cancelled
      || audioLoadStates.get(audio) !== state
    ) {
      return;
    }


    state.attempt += 1;


    /*
     * Remove listeners from the previous attempt.
     */
    if (state.controller) {
      state.controller.abort();
    }


    const controller =
      new AbortController();

    state.controller =
      controller;


    let settled = false;


    const clearAttempt = () => {
      if (state.timeoutId !== null) {
        clearTimeout(
          state.timeoutId
        );

        state.timeoutId = null;
      }

      /*
       * Removes loadeddata/error listeners registered
       * for this individual attempt.
       */
      controller.abort();
    };


    const succeed = () => {
      if (settled) {
        return;
      }

      settled = true;

      clearAttempt();

      if (
        audioLoadStates.get(audio)
          === state
      ) {
        audioLoadStates.delete(
          audio
        );
      }
    };


    const retry = reason => {
      if (settled) {
        return;
      }

      settled = true;

      clearAttempt();


      /*
       * Do not retry if this audio element has subsequently
       * been assigned to another trial.
       */
      if (
        state.cancelled
        || audioLoadStates.get(audio) !== state
      ) {
        return;
      }


      console.warn(
        `Audio load attempt ${state.attempt} failed `
        + `(${reason}). Retrying: ${url}`
      );


      /*
       * Retry the SAME URL indefinitely.
       */
      state.retryId =
        setTimeout(
          () => {
            state.retryId = null;

            tryLoad();
          },
          AUDIO_RETRY_DELAY_MS
        );
    };


    /*
     * Successful load.
     */
    audio.addEventListener(
      "loadeddata",
      succeed,
      {
        once: true,
        signal: controller.signal
      }
    );


    /*
     * Browser-reported media error.
     */
    audio.addEventListener(
      "error",
      () => {
        retry(
          "media error"
        );
      },
      {
        once: true,
        signal: controller.signal
      }
    );


    /*
     * Assign exactly the same URL on every retry.
     */
    audio.src = url;

    audio.load();


    /*
     * Handles cases where loading hangs without producing
     * either loadeddata or error.
     */
    state.timeoutId =
      setTimeout(
        () => {
          retry(
            `load timeout after `
            + `${AUDIO_LOAD_TIMEOUT_MS} ms`
          );
        },
        AUDIO_LOAD_TIMEOUT_MS
      );
  };


  tryLoad();
}



/*
 * ------------------------------------------------------------
 * Headphone check
 * ------------------------------------------------------------
 */

function renderHeadphoneCheck() {
  const container =
    document.getElementById(
      "headphoneTrials"
    );

  container.replaceChildren();


  HEADPHONE_CHECKS.forEach(
    (trial, trialIndex) => {

      headphonePlayCounts[
        trial.id
      ] = [0, 0, 0];


      const wrapper =
        document.createElement(
          "div"
        );

      wrapper.className =
        "headphone-trial";


      const heading =
        document.createElement(
          "h2"
        );

      heading.textContent =
        `Check ${trialIndex + 1} `
        + `of ${HEADPHONE_CHECKS.length}`;


      const options =
        document.createElement(
          "div"
        );

      options.className =
        "headphone-options";


      trial.urls.forEach(
        (
          url,
          positionIndex
        ) => {

          const option =
            document.createElement(
              "div"
            );

          option.className =
            "headphone-option";


          const title =
            document.createElement(
              "p"
            );

          title.textContent =
            `Sample ${positionIndex + 1}`;


          const audio =
            document.createElement(
              "audio"
            );

          audio.controls = true;

          audio.preload = "auto";


          audio.addEventListener(
            "play",
            () => {
              headphonePlayCounts[
                trial.id
              ][positionIndex] += 1;
            }
          );


          /*
           * Load using retry-until-success behavior.
           */
          loadAudioWithRetry(
            audio,
            url
          );


          const radioLabel =
            document.createElement(
              "label"
            );


          const radio =
            document.createElement(
              "input"
            );

          radio.type =
            "radio";

          radio.name =
            trial.id;

          radio.value =
            String(
              positionIndex + 1
            );


          radio.addEventListener(
            "change",
            () => {
              headphoneAnswers[
                trial.id
              ] = Number(
                radio.value
              );
            }
          );


          radioLabel.append(
            radio,
            ` Sample `
            + `${positionIndex + 1} `
            + `is quietest`
          );


          option.append(
            title,
            audio,
            radioLabel
          );


          options.appendChild(
            option
          );
        }
      );


      wrapper.append(
        heading,
        options
      );


      container.appendChild(
        wrapper
      );
    }
  );
}



/*
 * ------------------------------------------------------------
 * CSV loading
 * ------------------------------------------------------------
 */

async function parseCsv(path) {
  const response =
    await fetch(
      `${path}?v=`
      + encodeURIComponent(
        EXPERIMENT_VERSION
      ),
      {
        cache: "no-store"
      }
    );


  if (!response.ok) {
    throw new Error(
      `Could not load ${path} `
      + `(HTTP ${response.status}).`
    );
  }


  const text =
    await response.text();


  const parsed =
    Papa.parse(
      text,
      {
        header: true,
        skipEmptyLines: true,
        transformHeader:
          value => value.trim()
      }
    );


  if (parsed.errors.length) {
    console.error(
      parsed.errors
    );

    throw new Error(
      `${path} contains invalid CSV rows.`
    );
  }


  return parsed.data;
}



/*
 * ------------------------------------------------------------
 * Trial loading
 * ------------------------------------------------------------
 */

async function loadTrials() {
  const assignedSchedule =
    await claimParticipantSchedule();


  const scheduleRoot =
    `${GENERATED_TRIAL_ROOT}/`
    + `participant_schedules/`
    + assignedSchedule;


  const [
    triangleRows,
    preferenceRows
  ] = await Promise.all([
    parseCsv(
      `${scheduleRoot}/triangle_trials.csv`
    ),

    parseCsv(
      `${scheduleRoot}/preference_trials.csv`
    )
  ]);


  /*
   * Triangle trials
   */
  triangleTrials =
    triangleRows
      .map(
        row => {

          const canonicalSystemA =
            row.system_x
            || row.system_a;


          const canonicalSystemB =
            row.system_y
            || row.system_b;


          const sequence =
            String(
              row.sequence
              || ""
            ).trim();


          const systemForLetter =
            letter => {

              if (letter === "A") {
                return canonicalSystemA;
              }

              if (letter === "B") {
                return canonicalSystemB;
              }

              return null;
            };


          return {
            ...row,

            system_x:
              canonicalSystemA,

            system_y:
              canonicalSystemB,

            system_1:
              row.system_1
              || systemForLetter(
                sequence[0]
              ),

            system_2:
              row.system_2
              || systemForLetter(
                sequence[1]
              ),

            system_3:
              row.system_3
              || systemForLetter(
                sequence[2]
              ),

            audio_1_url:
              row.audio_1_url
              || row.audio_1,

            audio_2_url:
              row.audio_2_url
              || row.audio_2,

            audio_3_url:
              row.audio_3_url
              || row.audio_3,

            correct_position:
              Number(
                row.correct_position
              )
          };
        }
      )
      .filter(
        row =>
          row.trial_id
          && row.dataset
          && row.utterance_id
          && row.system_x
          && row.system_y
          && row.system_1
          && row.system_2
          && row.system_3
          && row.audio_1_url
          && row.audio_2_url
          && row.audio_3_url
          && [
            "AAB",
            "ABA",
            "BAA",
            "BBA",
            "BAB",
            "ABB"
          ].includes(
            row.sequence
          )
          && [
            1,
            2,
            3
          ].includes(
            row.correct_position
          )
      );


  /*
   * Preference trials
   */
  preferenceTrials =
    preferenceRows
      .map(
        row => {

          const canonicalSystemA =
            row.system_x
            || row.system_a;


          const canonicalSystemB =
            row.system_y
            || row.system_b;


          const presentationOrder =
            String(
              row.presentation_order
              || "AB"
            ).trim();


          const displayedSystemA =
            row.left_system
            || (
              presentationOrder === "BA"
                ? canonicalSystemB
                : canonicalSystemA
            );


          const displayedSystemB =
            row.right_system
            || (
              presentationOrder === "BA"
                ? canonicalSystemA
                : canonicalSystemB
            );


          return {
            ...row,

            system_x:
              canonicalSystemA,

            system_y:
              canonicalSystemB,

            system_a:
              displayedSystemA,

            system_b:
              displayedSystemB,

            audio_a_url:
              row.audio_a_url
              || row.audio_1,

            audio_b_url:
              row.audio_b_url
              || row.audio_2,

            presentation_order:
              presentationOrder
          };
        }
      )
      .filter(
        row =>
          row.trial_id
          && row.dataset
          && row.utterance_id
          && row.system_x
          && row.system_y
          && row.system_a
          && row.system_b
          && row.audio_a_url
          && row.audio_b_url
          && [
            "AB",
            "BA"
          ].includes(
            row.presentation_order
          )
      );


  console.log(
    `Loaded ${triangleTrials.length} `
    + `valid triangle trials `
    + `for ${assignedSchedule}.`
  );


  console.log(
    `Loaded ${preferenceTrials.length} `
    + `valid preference trials `
    + `for ${assignedSchedule}.`
  );


  if (
    triangleTrials.length
      < TRIANGLE_TRIAL_COUNT
  ) {
    console.error(
      "Rejected triangle rows:",
      triangleRows
    );

    throw new Error(
      `${assignedSchedule}/triangle_trials.csv `
      + `contains ${triangleTrials.length} `
      + `valid trials, but `
      + `${TRIANGLE_TRIAL_COUNT} `
      + `are required.`
    );
  }


  if (
    preferenceTrials.length
      < PREFERENCE_TRIAL_COUNT
  ) {
    console.error(
      "Rejected preference rows:",
      preferenceRows
    );

    throw new Error(
      `${assignedSchedule}/preference_trials.csv `
      + `contains ${preferenceTrials.length} `
      + `valid trials, but `
      + `${PREFERENCE_TRIAL_COUNT} `
      + `are required.`
    );
  }


  /*
   * Exactly the assigned trials are retained.
   *
   * Audio failures do not affect this list.
   */
  triangleTrials =
    triangleTrials.slice(
      0,
      TRIANGLE_TRIAL_COUNT
    );


  preferenceTrials =
    preferenceTrials.slice(
      0,
      PREFERENCE_TRIAL_COUNT
    );
}



/*
 * ------------------------------------------------------------
 * Stop currently active experiment audio
 * ------------------------------------------------------------
 */

function stopAllAudio() {
  [
    ...triangleAudio,
    ...preferenceAudio
  ].forEach(
    audio => {

      /*
       * Prevent retry processes belonging to the previous
       * trial from continuing after moving forward.
       */
      cancelPendingAudioLoad(
        audio
      );

      audio.pause();

      audio.currentTime = 0;
    }
  );
}



/*
 * ------------------------------------------------------------
 * Block instructions
 * ------------------------------------------------------------
 */

function showBlockInstructions(task) {
  currentTask = task;


  const title =
    document.getElementById(
      "instructionsTitle"
    );


  const body =
    document.getElementById(
      "instructionsBody"
    );


  if (task === "triangle") {

    title.textContent =
      "Discrimination task";


    body.innerHTML = `
      <p>
        This task contains ${TRIANGLE_TRIAL_COUNT} trials.
        Each trial has three recordings. Two are identical and one is
        different.
      </p>

      <p>
        Listen to all three recordings and identify the different one.
        No correctness feedback will be shown.
      </p>
    `;

  } else {

    title.textContent =
      "Preference task";


    body.innerHTML = `
      <p>
        This task contains ${PREFERENCE_TRIAL_COUNT} trials.
        Each trial has two recordings.
      </p>

      <p>
        Choose which recording you would prefer for everyday listening.
        Select “No preference” when neither recording is preferred.
      </p>
    `;
  }


  showOnly(
    stages.instructions
  );
}



/*
 * ------------------------------------------------------------
 * Start block
 * ------------------------------------------------------------
 */

function beginCurrentBlock() {
  if (
    currentTask === "triangle"
  ) {
    showOnly(
      stages.triangle
    );

    showTriangleTrial();

  } else {

    showOnly(
      stages.preference
    );

    showPreferenceTrial();
  }
}



/*
 * ------------------------------------------------------------
 * Complete block
 * ------------------------------------------------------------
 */

async function completeCurrentBlock() {
  currentBlockPosition += 1;


  const experimentCompleted =
    currentBlockPosition
      >= blockOrder.length;


  try {
    await upsertSession(
      experimentCompleted
        ? "completed"
        : "between_blocks"
    );
  } catch (error) {
    console.error(
      "Could not update block completion status:",
      error
    );
  }


  if (experimentCompleted) {
    showOnly(
      stages.done
    );

    return;
  }


  showOnly(
    stages.break
  );
}



/*
 * ------------------------------------------------------------
 * Triangle task
 * ------------------------------------------------------------
 */

function showTriangleTrial() {
  stopAllAudio();


  if (
    triangleIndex
      >= TRIANGLE_TRIAL_COUNT
  ) {
    void completeCurrentBlock();
    return;
  }


  const trial =
    triangleTrials[
      triangleIndex
    ];


  saving = false;


  trianglePlayCounts =
    [0, 0, 0];


  triangleHasPlayed =
    [false, false, false];


  document.getElementById(
    "triangleError"
  ).textContent = "";


  document.getElementById(
    "triangleCounter"
  ).textContent =
    `Trial ${triangleIndex + 1} `
    + `of ${TRIANGLE_TRIAL_COUNT}`;


  const progress =
    document.getElementById(
      "triangleProgress"
    );


  progress.max =
    TRIANGLE_TRIAL_COUNT;


  progress.value =
    triangleIndex;


  const urls = [
    trial.audio_1_url,
    trial.audio_2_url,
    trial.audio_3_url
  ];


  triangleAudio.forEach(
    (audio, index) => {

      /*
       * Each file retries independently until it loads.
       *
       * The same URL is always used.
       */
      loadAudioWithRetry(
        audio,
        urls[index]
      );
    }
  );


  updateTriangleAvailability();


  trialStartMs =
    performance.now();
}



function updateTriangleAvailability() {
  const ready =
    triangleHasPlayed.every(
      Boolean
    );


  triangleButtons.forEach(
    button => {
      button.disabled =
        !ready
        || saving;
    }
  );


  document.getElementById(
    "triangleStatus"
  ).textContent =
    ready
      ? "Select the different recording."
      : "Listen to all three recordings before answering.";
}



async function saveTriangleResponse(
  selectedPosition
) {
  if (
    saving
    || !triangleHasPlayed.every(Boolean)
  ) {
    return;
  }


  saving = true;


  updateTriangleAvailability();


  const trial =
    triangleTrials[
      triangleIndex
    ];


  const correctPosition =
    Number(
      trial.correct_position
    );


  const row = {
    participant_id:
      participantId,

    session_id:
      sessionId,

    experiment_version:
      EXPERIMENT_VERSION,

    schedule_id:
      scheduleId,

    task:
      "triangle",

    block_position:
      currentBlockPosition + 1,

    trial_id:
      trial.trial_id,

    dataset:
      trial.dataset,

    utterance_id:
      trial.utterance_id,

    system_x:
      trial.system_x,

    system_y:
      trial.system_y,

    sequence:
      trial.sequence,

    system_1:
      trial.system_1,

    system_2:
      trial.system_2,

    system_3:
      trial.system_3,

    selected_position:
      selectedPosition,

    correct_position:
      correctPosition,

    is_correct:
      selectedPosition
        === correctPosition,

    response_time_ms:
      Math.round(
        performance.now()
        - trialStartMs
      ),

    play_count_1:
      trianglePlayCounts[0],

    play_count_2:
      trianglePlayCounts[1],

    play_count_3:
      trianglePlayCounts[2],

    trial_index:
      triangleIndex + 1
  };


  const { error } =
    await db
      .from(
        "triangle_responses"
      )
      .insert(
        row
      );


  if (error) {
    console.error(
      error
    );


    document.getElementById(
      "triangleError"
    ).textContent =
      `The response could not be saved: `
      + error.message;


    saving = false;


    updateTriangleAvailability();


    return;
  }


  triangleIndex += 1;


  showTriangleTrial();
}



/*
 * ------------------------------------------------------------
 * Preference task
 * ------------------------------------------------------------
 */

function showPreferenceTrial() {
  stopAllAudio();


  if (
    preferenceIndex
      >= PREFERENCE_TRIAL_COUNT
  ) {
    void completeCurrentBlock();
    return;
  }


  const trial =
    preferenceTrials[
      preferenceIndex
    ];


  saving = false;


  preferencePlayCounts =
    [0, 0];


  preferenceHasPlayed =
    [false, false];


  document.getElementById(
    "preferenceError"
  ).textContent = "";


  document.getElementById(
    "preferenceCounter"
  ).textContent =
    `Trial ${preferenceIndex + 1} `
    + `of ${PREFERENCE_TRIAL_COUNT}`;


  const progress =
    document.getElementById(
      "preferenceProgress"
    );


  progress.max =
    PREFERENCE_TRIAL_COUNT;


  progress.value =
    preferenceIndex;


  /*
   * Both recordings retry independently until successful.
   */
  loadAudioWithRetry(
    preferenceAudio[0],
    trial.audio_a_url
  );


  loadAudioWithRetry(
    preferenceAudio[1],
    trial.audio_b_url
  );


  updatePreferenceAvailability();


  trialStartMs =
    performance.now();
}



function updatePreferenceAvailability() {
  const ready =
    preferenceHasPlayed.every(
      Boolean
    );


  preferenceButtons.forEach(
    button => {
      button.disabled =
        !ready
        || saving;
    }
  );


  document.getElementById(
    "noPreferenceButton"
  ).disabled =
    !ready
    || saving;


  document.getElementById(
    "preferenceStatus"
  ).textContent =
    ready
      ? "Select your preferred recording, or choose no preference."
      : "Listen to both recordings before answering.";
}



async function savePreferenceResponse(
  choice
) {
  if (
    saving
    || !preferenceHasPlayed.every(Boolean)
  ) {
    return;
  }


  saving = true;


  updatePreferenceAvailability();


  const trial =
    preferenceTrials[
      preferenceIndex
    ];


  let preferredSystem =
    null;


  let noPreference =
    false;


  if (choice === "A") {

    preferredSystem =
      trial.system_a;

  } else if (
    choice === "B"
  ) {

    preferredSystem =
      trial.system_b;

  } else {

    noPreference =
      true;
  }


  const row = {
    participant_id:
      participantId,

    session_id:
      sessionId,

    experiment_version:
      EXPERIMENT_VERSION,

    schedule_id:
      scheduleId,

    task:
      "preference",

    block_position:
      currentBlockPosition + 1,

    trial_id:
      trial.trial_id,

    dataset:
      trial.dataset,

    utterance_id:
      trial.utterance_id,

    system_x:
      trial.system_x,

    system_y:
      trial.system_y,

    system_a:
      trial.system_a,

    system_b:
      trial.system_b,

    presentation_order:
      trial.presentation_order,

    selected_option:
      choice,

    preferred_system:
      preferredSystem,

    no_preference:
      noPreference,

    response_time_ms:
      Math.round(
        performance.now()
        - trialStartMs
      ),

    play_count_a:
      preferencePlayCounts[0],

    play_count_b:
      preferencePlayCounts[1],

    trial_index:
      preferenceIndex + 1
  };


  const { error } =
    await db
      .from(
        "preference_responses"
      )
      .insert(
        row
      );


  if (error) {
    console.error(
      error
    );


    document.getElementById(
      "preferenceError"
    ).textContent =
      `The response could not be saved: `
      + error.message;


    saving = false;


    updatePreferenceAvailability();


    return;
  }


  preferenceIndex += 1;


  showPreferenceTrial();
}



/*
 * ------------------------------------------------------------
 * Consent -> headphone check
 * ------------------------------------------------------------
 */

document.getElementById(
  "toHeadphoneCheck"
).addEventListener(
  "click",
  () => {

    const consent =
      document.getElementById(
        "consent"
      ).checked;


    const adult =
      document.getElementById(
        "adult"
      ).checked;


    const quietRoom =
      document.getElementById(
        "quietRoom"
      ).checked;


    const headphones =
      document.getElementById(
        "headphones"
      ).checked;


    const deviceType =
      document.getElementById(
        "deviceType"
      ).value;


    const environmentNoise =
      document.getElementById(
        "environmentNoise"
      ).value;


    const error =
      document.getElementById(
        "consentError"
      );


    error.textContent = "";


    if (
      !consent
      || !adult
      || !quietRoom
      || !headphones
    ) {
      error.textContent =
        "All confirmations are required before continuing.";

      return;
    }


    if (
      !deviceType
      || !environmentNoise
    ) {
      error.textContent =
        "Select your listening device and background-noise level.";

      return;
    }


    environmentMetadata = {
      consent_given:
        consent,

      adult_confirmed:
        adult,

      quiet_room_confirmed:
        quietRoom,

      headphones_confirmed:
        headphones,

      device_type:
        deviceType,

      environment_noise:
        environmentNoise
    };


    collectParticipantMetadata();


    renderHeadphoneCheck();


    showOnly(
      stages.headphone
    );
  }
);



/*
 * ------------------------------------------------------------
 * Submit headphone check
 * ------------------------------------------------------------
 */

document.getElementById(
  "submitHeadphoneCheck"
).addEventListener(
  "click",
  async () => {

    const error =
      document.getElementById(
        "headphoneError"
      );


    error.textContent = "";


    if (
      Object.keys(
        headphoneAnswers
      ).length
        !== HEADPHONE_CHECKS.length
    ) {
      error.textContent =
        "Answer all headphone-check trials.";

      return;
    }


    const allPlayed =
      HEADPHONE_CHECKS.every(
        trial =>
          headphonePlayCounts[
            trial.id
          ].every(
            count =>
              count > 0
          )
      );


    if (!allPlayed) {
      error.textContent =
        "Play every headphone-check sample before submitting.";

      return;
    }


    const correct =
      HEADPHONE_CHECKS.reduce(
        (
          sum,
          trial
        ) => {

          return (
            sum
            + (
              headphoneAnswers[
                trial.id
              ]
                === trial.correctPosition
                ? 1
                : 0
            )
          );
        },
        0
      );


    headphonePassed =
      correct >= 2;


    chooseBlockOrder();


    try {

      if (headphonePassed) {
        await claimParticipantSchedule();
      }


      await upsertSession(
        headphonePassed
          ? "headphone_passed"
          : "headphone_failed"
      );

    } catch (saveError) {

      error.textContent =
        `Could not save the screening result: `
        + saveError.message;

      return;
    }


    showOnly(
      headphonePassed
        ? stages.calibration
        : stages.fail
    );
  }
);



/*
 * ------------------------------------------------------------
 * Start experiment
 * ------------------------------------------------------------
 */

document.getElementById(
  "startExperiment"
).addEventListener(
  "click",
  async () => {

    const error =
      document.getElementById(
        "calibrationError"
      );


    error.textContent = "";


    if (
      !document.getElementById(
        "volumeConfirmed"
      ).checked
    ) {
      error.textContent =
        "Confirm that the calibration sample is comfortably audible.";

      return;
    }


    try {

      await loadTrials();


      await upsertSession(
        "started"
      );

    } catch (loadError) {

      console.error(
        loadError
      );


      error.textContent =
        loadError.message;


      return;
    }


    currentBlockPosition = 0;


    showBlockInstructions(
      blockOrder[
        currentBlockPosition
      ]
    );
  }
);



/*
 * ------------------------------------------------------------
 * Begin first/current block
 * ------------------------------------------------------------
 */

document.getElementById(
  "beginBlock"
).addEventListener(
  "click",
  beginCurrentBlock
);



/*
 * ------------------------------------------------------------
 * Begin second block
 * ------------------------------------------------------------
 */

document.getElementById(
  "startSecondBlock"
).addEventListener(
  "click",
  () => {

    showBlockInstructions(
      blockOrder[
        currentBlockPosition
      ]
    );
  }
);



/*
 * ------------------------------------------------------------
 * Triangle audio playback
 * ------------------------------------------------------------
 */

triangleAudio.forEach(
  (
    audio,
    index
  ) => {

    audio.addEventListener(
      "play",
      () => {

        trianglePlayCounts[
          index
        ] += 1;


        triangleHasPlayed[
          index
        ] = true;


        /*
         * Only one recording plays at a time.
         */
        triangleAudio.forEach(
          (
            other,
            otherIndex
          ) => {

            if (
              otherIndex !== index
            ) {
              other.pause();
            }
          }
        );


        updateTriangleAvailability();
      }
    );
  }
);



/*
 * ------------------------------------------------------------
 * Preference audio playback
 * ------------------------------------------------------------
 */

preferenceAudio.forEach(
  (
    audio,
    index
  ) => {

    audio.addEventListener(
      "play",
      () => {

        preferencePlayCounts[
          index
        ] += 1;


        preferenceHasPlayed[
          index
        ] = true;


        /*
         * Only one recording plays at a time.
         */
        preferenceAudio.forEach(
          (
            other,
            otherIndex
          ) => {

            if (
              otherIndex !== index
            ) {
              other.pause();
            }
          }
        );


        updatePreferenceAvailability();
      }
    );
  }
);



/*
 * ------------------------------------------------------------
 * Triangle response buttons
 * ------------------------------------------------------------
 */

triangleButtons.forEach(
  button => {

    button.addEventListener(
      "click",
      () => {

        void saveTriangleResponse(
          Number(
            button.dataset.position
          )
        );
      }
    );
  }
);



/*
 * ------------------------------------------------------------
 * Preference response buttons
 * ------------------------------------------------------------
 */

preferenceButtons.forEach(
  button => {

    button.addEventListener(
      "click",
      () => {

        void savePreferenceResponse(
          button.dataset.choice
        );
      }
    );
  }
);



/*
 * ------------------------------------------------------------
 * No-preference button
 * ------------------------------------------------------------
 */

document.getElementById(
  "noPreferenceButton"
).addEventListener(
  "click",
  () => {

    void savePreferenceResponse(
      "none"
    );
  }
);