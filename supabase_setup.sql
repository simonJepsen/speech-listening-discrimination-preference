-- WARNING:
-- This script drops and recreates the study tables.
-- Export existing data before running it.

drop table if exists public.preference_responses cascade;
drop table if exists public.triangle_responses cascade;
drop table if exists public.sessions cascade;
drop table if exists public.schedule_assignments cascade;

create table public.sessions (
    id bigint generated always as identity primary key,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),

    participant_id text not null,
    session_id text not null unique,
    experiment_version text not null,
    schedule_id text,

    status text not null check (
        status in (
            'headphone_passed',
            'headphone_failed',
            'started',
            'between_blocks',
            'completed'
        )
    ),

    consent_given boolean not null,
    adult_confirmed boolean not null,
    quiet_room_confirmed boolean not null,
    headphones_confirmed boolean not null,

    device_type text,
    environment_noise text,

    age_group text,
    sex_assigned_at_birth text,
    gender_identity text,
    native_language text,
    english_proficiency text,
    english_use_frequency text,
    normal_hearing boolean,
    hearing_aid_use text,
    headphone_type text,
    listening_environment text,
    browser text,
    operating_system text,

    headphone_passed boolean not null default false,
    headphone_correct integer not null default 0,
    headphone_total integer not null default 0,

    block_order text,
    triangle_completed integer not null default 0,
    preference_completed integer not null default 0,

    user_agent text
);


-- Keep sessions.updated_at current when the frontend upserts session state.
create or replace function public.set_updated_at()
returns trigger
language plpgsql
set search_path = public
as $$
begin
    new.updated_at = now();
    return new;
end;
$$;

create trigger sessions_set_updated_at
before update on public.sessions
for each row
execute function public.set_updated_at();

create table public.triangle_responses (
    id bigint generated always as identity primary key,
    created_at timestamptz not null default now(),

    participant_id text not null,
    session_id text not null,
    experiment_version text not null,
    schedule_id text not null,
    task text not null default 'triangle'
        check (task = 'triangle'),
    block_position smallint not null
        check (block_position in (1, 2)),

    trial_id text not null,
    dataset text not null,
    utterance_id text not null,

    system_x text not null,
    system_y text not null,
    sequence text not null,

    system_1 text not null,
    system_2 text not null,
    system_3 text not null,

    selected_position smallint not null
        check (selected_position between 1 and 3),
    correct_position smallint not null
        check (correct_position between 1 and 3),
    is_correct boolean not null,

    response_time_ms integer not null
        check (response_time_ms >= 0),

    play_count_1 integer not null
        check (play_count_1 > 0),
    play_count_2 integer not null
        check (play_count_2 > 0),
    play_count_3 integer not null
        check (play_count_3 > 0),

    trial_index integer not null
        check (trial_index between 1 and 30),

    unique (session_id, trial_id)
);

create table public.preference_responses (
    id bigint generated always as identity primary key,
    created_at timestamptz not null default now(),

    participant_id text not null,
    session_id text not null,
    experiment_version text not null,
    schedule_id text not null,
    task text not null default 'preference'
        check (task = 'preference'),
    block_position smallint not null
        check (block_position in (1, 2)),

    trial_id text not null,
    dataset text not null,
    utterance_id text not null,

    system_x text not null,
    system_y text not null,

    system_a text not null,
    system_b text not null,
    presentation_order text not null
        check (presentation_order in ('AB', 'BA')),

    selected_option text not null
        check (selected_option in ('A', 'B', 'none')),
    preferred_system text,
    no_preference boolean not null,

    response_time_ms integer not null
        check (response_time_ms >= 0),

    play_count_a integer not null
        check (play_count_a > 0),
    play_count_b integer not null
        check (play_count_b > 0),

    trial_index integer not null
        check (trial_index between 1 and 30),

    check (
        (
            selected_option = 'none'
            and no_preference = true
            and preferred_system is null
        )
        or
        (
            selected_option in ('A', 'B')
            and no_preference = false
            and preferred_system is not null
        )
    ),

    unique (session_id, trial_id)
);

create table public.schedule_assignments (
    id bigint generated always as identity primary key,
    created_at timestamptz not null default now(),
    participant_id text not null,
    experiment_version text not null,
    schedule_number integer not null check (schedule_number > 0),
    unique (participant_id, experiment_version)
);

create index schedule_assignment_balance_idx
on public.schedule_assignments (experiment_version, schedule_number);

-- Atomically return the participant's existing schedule or assign one of the
-- currently least-used schedules. The advisory lock prevents two simultaneous
-- participants from being allocated from the same stale count snapshot.
create or replace function public.claim_listening_schedule(
    p_participant_id text,
    p_experiment_version text,
    p_schedule_count integer
)
returns integer
language plpgsql
security definer
set search_path = public
as $$
declare
    assigned integer;
begin
    if p_participant_id is null or length(p_participant_id) < 8 then
        raise exception 'Invalid participant ID';
    end if;
    if p_schedule_count < 1 or p_schedule_count > 10000 then
        raise exception 'Invalid schedule count';
    end if;

    perform pg_advisory_xact_lock(hashtext(p_experiment_version));

    select schedule_number into assigned
    from public.schedule_assignments
    where participant_id = p_participant_id
      and experiment_version = p_experiment_version;

    if assigned is not null then
        return assigned;
    end if;

    select candidate.schedule_number into assigned
    from generate_series(1, p_schedule_count) as candidate(schedule_number)
    left join (
        select schedule_number, count(*) as uses
        from public.schedule_assignments
        where experiment_version = p_experiment_version
        group by schedule_number
    ) counts using (schedule_number)
    order by coalesce(counts.uses, 0), random()
    limit 1;

    insert into public.schedule_assignments (
        participant_id, experiment_version, schedule_number
    ) values (
        p_participant_id, p_experiment_version, assigned
    );

    return assigned;
end;
$$;

revoke all on function public.claim_listening_schedule(text, text, integer) from public;
grant execute on function public.claim_listening_schedule(text, text, integer) to anon;

create index triangle_pair_idx
on public.triangle_responses (system_x, system_y);

create index triangle_utterance_idx
on public.triangle_responses (utterance_id);

create index preference_pair_idx
on public.preference_responses (system_x, system_y);

create index preference_utterance_idx
on public.preference_responses (utterance_id);


-- Permissions used by the browser client. Row-level security policies below
-- still determine which operations are permitted.
grant insert, update on public.sessions to anon;
grant insert on public.triangle_responses to anon;
grant insert on public.preference_responses to anon;


grant usage, select on sequence public.sessions_id_seq to anon;
grant usage, select on sequence public.triangle_responses_id_seq to anon;
grant usage, select on sequence public.preference_responses_id_seq to anon;

alter table public.schedule_assignments enable row level security;
alter table public.sessions enable row level security;
alter table public.triangle_responses enable row level security;
alter table public.preference_responses enable row level security;

create policy "Anonymous session insert"
on public.sessions
for insert
to anon
with check (
    consent_given = true
    and adult_confirmed = true
    and quiet_room_confirmed = true
    and headphones_confirmed = true
);

create policy "Anonymous session update"
on public.sessions
for update
to anon
using (true)
with check (
    consent_given = true
    and adult_confirmed = true
    and quiet_room_confirmed = true
    and headphones_confirmed = true
);

create policy "Anonymous triangle insert"
on public.triangle_responses
for insert
to anon
with check (
    task = 'triangle'
    and selected_position between 1 and 3
    and correct_position between 1 and 3
    and is_correct = (selected_position = correct_position)
    and response_time_ms >= 0
    and play_count_1 > 0
    and play_count_2 > 0
    and play_count_3 > 0
);

create policy "Anonymous preference insert"
on public.preference_responses
for insert
to anon
with check (
    task = 'preference'
    and selected_option in ('A', 'B', 'none')
    and response_time_ms >= 0
    and play_count_a > 0
    and play_count_b > 0
);
