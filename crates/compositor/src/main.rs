pub mod backend;
pub mod colors;
pub mod config;

use std::{sync::Arc, task::Poll, time::Duration};

// use skia_safe::prelude::NativeAccess;

use smithay::{
    reexports::{
        calloop,
        drm::{self, Device, control::Device as _},
        gbm,
    },
    utils::DevPath,
};

use crate::backend::tty::{
    self, BufferedOutput,
    cpu::Cpu,
    drmx::{self, DrmEventNotifier},
};

fn main() -> anyhow::Result<()> {
    pretty_env_logger::init();

    let card = tty::drmx::Card::find_primary()?;

    log::trace!("DRM Driver: {:?}", card.get_driver());
    let resources = card.resource_handles()?;

    let card = Arc::new(gbm::Device::new(card)?);

    log::trace!("Using card {:?}", card.dev_path());

    // Pick a suitable CRTC and mode for the active connector.
    let connector = resources
        .connectors()
        .iter()
        .filter_map(|&con| card.get_connector(con, true).ok())
        // TODO: Make this slightly more configurable...
        .find(|con| con.state() == drm::control::connector::State::Connected)
        .ok_or(anyhow::anyhow!("Not connected to any display!"))?;

    let crtc = {
        let compatible_crtcs = connector
            .encoders()
            .iter()
            .filter_map(|&enc| card.get_encoder(enc).ok())
            .flat_map(|enc| resources.filter_crtcs(enc.possible_crtcs()))
            .collect::<Vec<_>>();

        let crtc_before = connector
            .current_encoder()
            .and_then(|enc| card.get_encoder(enc).ok())
            .and_then(|encoder| encoder.crtc());
        compatible_crtcs
            .iter()
            .find(|&crtc| crtc_before.as_ref().is_some_and(|old| old == crtc))
            .or(compatible_crtcs.first())
            .copied()
            .ok_or(anyhow::anyhow!("Cannot find a suitable CRTC!"))?
    };

    let mode = {
        let modes = connector.modes();
        modes
            .iter()
            .find(|&mode| {
                mode.mode_type()
                    .contains(drm::control::ModeTypeFlags::PREFERRED)
            })
            .or(modes.first())
            .copied()
            .ok_or(anyhow::anyhow!(
                "Cannot find suitable mode for connector {}",
                connector.interface().as_str()
            ))?
    };

    let planes = card
        .plane_handles()?
        .iter()
        .filter_map(|&p| card.get_plane(p).ok())
        .filter(|p| resources.filter_crtcs(p.possible_crtcs()).contains(&crtc))
        .collect::<Vec<_>>();

    let (plane, _plane_props) = planes
        .iter()
        .filter_map(|p| card.get_properties(p.handle()).ok().map(|info| (p, info)))
        .filter_map(|(p, info)| {
            let ty = info.iter().find_map(|(k, v)| {
                if let Ok(info) = card.get_property(*k)
                    && info.name() == c"type"
                {
                    use drm::control::PlaneType as Type;

                    const PRIMARY: u64 = Type::Primary as u64;
                    const CURSOR: u64 = Type::Cursor as u64;
                    const OVERLAY: u64 = Type::Overlay as u64;
                    return match *v {
                        PRIMARY => Some(Type::Primary),
                        CURSOR => Some(Type::Cursor),
                        OVERLAY => Some(Type::Overlay),
                        _ => None,
                    };
                }

                None
            });

            ty.map(move |ty| (p.handle(), ty, info))
        })
        .find_map(|(p, ty, props)| (ty == drm::control::PlaneType::Primary).then_some((p, props)))
        .ok_or(anyhow::anyhow!("cannot find suitable primary plane"))?;

    let format = drmx::Format(gbm::Format::Xrgb8888);

    // let mut skia = device.clone().skia_context()?;

    let mut output =
        BufferedOutput::new(card.clone(), crtc, plane, mode, connector.handle(), format)?;
    let mut event_loop = calloop::EventLoop::try_new()?;

    event_loop.handle().insert_source(
        DrmEventNotifier::new(card.clone()),
        |event, _, output: &mut BufferedOutput<Cpu>| {
            if let drm::control::Event::PageFlip(page_flip_event) = event {
                log::trace!(
                    "Flip event! {:?} @ {:?}",
                    page_flip_event.frame,
                    page_flip_event.duration
                );

                if let Err(err) = output.flip() {
                    log::error!("Error during page flip: {err:?}");
                }
            }
        },
    )?;

    let signal = event_loop.get_signal();

    event_loop
        .handle()
        .insert_source(
            calloop::timer::Timer::from_duration(Duration::from_secs(10)),
            move |_, _, _| {
                signal.stop();
                calloop::timer::TimeoutAction::Drop
            },
        )
        .map_err(|a| a.error)?;

    let Poll::Ready(()) = output.render()? else {
        panic!("First buffer is still being scanned! Should not be the case!")
    };

    log::trace!("Before Flip page!");
    let Poll::Ready(()) = output.flip()? else {
        panic!("Should not have a problem with queuing an initial flip.")
    };
    
    log::trace!("After Flip page!");
    let now = std::time::Instant::now();
    event_loop.run(
        Some(Duration::from_millis(5)),
        &mut output,
        |output| match output.render().expect("No error within render loop") {
            Poll::Ready(()) => (),
            Poll::Pending => (),
        },
    )?;

    let fps = output.frame as f64 / now.elapsed().as_secs_f64();
    println!("Average FPS: {fps:.3}");

    Ok(())
}
